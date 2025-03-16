from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange
from mlx_lm.models.base import scaled_dot_product_attention
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from csm_mlx.lm.config import ModelType
from csm_mlx.lm.utils.rope import Llama3RoPE


class RopeScaling(BaseModel):
    factor: float
    high_freq_factor: float
    low_freq_factor: float
    original_max_position_embeddings: int
    rope_type: str


class RQTransformerModelArgs(BaseModel):
    model_type: str

    # Base transformer trunk
    vocab_size: int
    n_layer: int
    n_head: int
    n_local_heads: int
    head_dim: int = 64
    dim: int
    intermediate_size: int
    rope_base: float = 10_000
    rope_scaling: Optional[RopeScaling]
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False

    # Fast layers
    codebook_size: int = 2048
    num_codebooks: int = 8
    n_fast_layer: int = 4
    fast_dim: int
    fast_n_head: int
    fast_n_local_heads: int
    fast_head_dim: int
    fast_intermediate_size: int
    fast_attention_qkv_bias: bool = False
    depthwise_wte: Optional[bool] = Field(default=None)
    depthwise_output: Optional[bool] = Field(default=None)
    duplicate_code_0: Optional[bool] = Field(default=True)

    # meta
    use_gradient_checkpointing: bool = False

    @classmethod
    def from_json_file(cls, file_path: str) -> "RQTransformerModelArgs":
        with open(file_path, "r") as f:
            return cls.model_validate_json(f.read())


class TokenConfig(BaseModel):
    im_end_id: int
    pad_id: int
    semantic_start_id: int
    semantic_end_id: Optional[int]

    @classmethod
    def from_tokenizer(
        cls, model: ModelType, tokenizer: Tokenizer, config: RQTransformerModelArgs
    ):
        im_end = tokenizer.token_to_id("<|im_end|>")
        if im_end is None:
            raise ValueError("Tokenizer does not have <|im_end|>")

        if model.family == "dual_ar" or (
            model.family == "fish" and model.version == "1.5"
        ):
            semantic_start_id = tokenizer.token_to_id("<|semantic:0|>")
        else:
            semantic_start_id = tokenizer.token_to_id("<|semantic|>") or 5

        semantic_end_id = None
        pad_id = tokenizer.token_to_id("<|semantic|>") or 5

        if model.family == "dual_ar" or (
            model.family == "fish" and model.version == "1.5"
        ):
            semantic_end_id = tokenizer.token_to_id(
                f"<|semantic:{config.codebook_size - 1}|>"
            )

        return cls(
            **{
                "im_end_id": im_end,
                "semantic_start_id": semantic_start_id,
                "semantic_end_id": semantic_end_id,
                "pad_id": pad_id,
            }
        )


class RQTransformer(nn.Module):
    def __init__(
        self,
        config: RQTransformerModelArgs,
        token_config: TokenConfig,
        model_type: ModelType,
    ):
        self.config = config
        self.token_config = token_config
        self.model_type = model_type

        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.max_fast_seqlen = config.num_codebooks - (
            0 if config.duplicate_code_0 else 1
        )
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks, config.dim
        )
        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        if not self.config.tie_word_embeddings:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        else:
            self.output = None

        if config.fast_dim != config.dim:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim)
        else:
            self.fast_project_in = nn.Identity()

        fast_embedding_input_dim = (
            config.codebook_size * (config.num_codebooks - 1)
            if config.depthwise_wte
            else config.codebook_size
        )
        if config.depthwise_wte:
            self.fast_embeddings = nn.Embedding(
                fast_embedding_input_dim, config.fast_dim
            )
        else:
            self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)

        self.fast_layers = [
            TransformerBlock(config, is_fast=True) for _ in range(config.n_fast_layer)
        ]
        self.fast_norm = nn.RMSNorm(config.fast_dim, eps=config.norm_eps)
        fast_output_dim = (
            config.codebook_size * self.max_fast_seqlen
            if config.depthwise_output
            else config.codebook_size
        )
        self.fast_output = nn.Linear(config.fast_dim, fast_output_dim, bias=False)
        self._semantic_offset = mx.arange(
            0,
            self.config.num_codebooks * self.config.codebook_size,
            self.config.codebook_size,
        )[:, mx.newaxis]

    def embed(self, x: mx.array) -> mx.array:
        semantic_tokens = x[:, 0, :]
        semantic_embeds = self.embeddings(semantic_tokens)[:, mx.newaxis, :]

        offset = (
            self._semantic_offset
            if self.config.duplicate_code_0
            else self._semantic_offset[1:, :]
        )
        codebook_tokens = x[:, 1:, :] + offset
        codebook_embeds = self.codebook_embeddings(codebook_tokens)

        if self.token_config.semantic_end_id is not None:
            emb_mask = (semantic_tokens >= self.token_config.semantic_start_id) & (
                semantic_tokens <= self.token_config.semantic_end_id
            )
        else:
            emb_mask = semantic_tokens == self.token_config.semantic_start_id

        codebook_embeds = codebook_embeds * emb_mask[:, :, mx.newaxis]
        return mx.concat([semantic_embeds, codebook_embeds], axis=1).sum(axis=1)

    # def __call__(self, ):
    def forward_generate(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> Tuple[mx.array, mx.array]:
        x = self.embed(inputs)
        mask = create_attention_mask(x, cache) if x.shape[1] > 1 else None

        for layer, layer_cache in zip(self.layers, cache or [None] * len(self.layers)):
            x = layer(x, mask=mask, cache=layer_cache)

        x = x[:, -1, :]  # Only take the last token for generation
        slow_out = self.norm(x)
        if self.output is not None:
            token_logits = self.output(slow_out)
        else:
            token_logits = self.embeddings.as_linear(slow_out)

        x = self.fast_project_in(x)
        return (token_logits, x)

    def forward_generate_fast(
        self,
        x: mx.array,
        input_pos: int,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """
        Assumes (bsz, seqlen=1, fast_dim)
        """
        mask = create_attention_mask(x, cache) if x.shape[1] > 1 else None

        for layer, layer_cache in zip(
            self.fast_layers, cache or [None] * len(self.fast_layers)
        ):
            x = layer(x, mask=mask, cache=layer_cache)

        fast_out = self.fast_norm(x)
        if self.config.depthwise_output:
            out_proj = self.fast_output.weight[
                input_pos * self.config.codebook_size : (input_pos + 1)
                * self.config.codebook_size,
                :,
            ]
            return fast_out @ out_proj.T

        else:
            return self.fast_output(fast_out)


class TransformerBlock(nn.Module):
    def __init__(self, config: RQTransformerModelArgs, is_fast: bool = False):
        super().__init__()
        self.attention = Attention(config, is_fast)
        self.feed_forward = MLP(config, is_fast)
        dim = config.fast_dim if is_fast else config.dim
        self.ffn_norm = nn.RMSNorm(dims=dim, eps=config.norm_eps)
        self.attention_norm = nn.RMSNorm(dim, config.norm_eps)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        h = self.attention_norm(x)
        attn = self.attention(h, mask=mask, cache=cache)
        h = x + attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: RQTransformerModelArgs, is_fast: bool = False):
        super().__init__()
        # GQA: groups split hidden dim evenly between them
        assert config.dim % config.n_head == 0
        dim = config.fast_dim if is_fast else config.dim
        n_head = config.fast_n_head if is_fast else config.n_head
        n_local_heads = config.fast_n_local_heads if is_fast else config.n_local_heads
        head_dim = config.fast_head_dim if is_fast else config.head_dim

        self.rope = (
            Llama3RoPE(
                dims=int(dim / n_head),
                max_position_embeddings=2048,
                traditional=True,
                base=int(config.rope_base),
                scaling_config={
                    "factor": config.rope_scaling.factor,
                    "low_freq_factor": config.rope_scaling.low_freq_factor,
                    "high_freq_factor": config.rope_scaling.high_freq_factor,
                    "old_context_len": config.rope_scaling.original_max_position_embeddings,
                },
            )
            if config.rope_scaling is not None
            else nn.RoPE(
                int(dim / n_head),
                traditional=False,
                base=config.rope_base,
            )
        )

        total_head_dim = (n_head + 2 * n_local_heads) * head_dim

        self.wqkv = nn.Linear(
            input_dims=dim,
            output_dims=total_head_dim,
            bias=config.attention_qkv_bias,
        )
        self.wo = nn.Linear(dim, dim, bias=False)

        self.n_local_heads, self.n_head, self.head_dim, self.dim = (
            n_local_heads,
            n_head,
            head_dim,
            dim,
        )
        # Manually apply $\sqrt{d_k}$
        self.scale = head_dim**-0.5

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        qkv = self.wqkv(x)
        qkv = rearrange(qkv, "b s (h d) -> b h s d", d=self.head_dim)
        q, k, v = qkv.split([self.n_head, self.n_head + self.n_local_heads], axis=1)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        output = scaled_dot_product_attention(
            queries=q,
            keys=k,
            values=v,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = rearrange(output, "b h s d -> b s (h d)")
        return self.wo(output)


class MLP(nn.Module):
    def __init__(self, config: RQTransformerModelArgs, is_fast=True) -> None:
        super().__init__()

        dim = config.fast_dim if is_fast else config.dim
        intermediate_size = (
            config.fast_intermediate_size if is_fast else config.intermediate_size
        )

        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


def create_attention_mask(h: mx.array, cache: Optional[Any] = None):
    """Creates an attention mask for the current decoding step.

    Args:
        h: Input tensor of shape [batch, seq_len, hidden_dim]
        cache: Optional KV cache containing previous key/value pairs

    Returns:
        A causal mask tensor or None if only processing a single token
    """
    query_seq_len = h.shape[1]
    if query_seq_len <= 1:
        return None

    # For history length, check the cache
    history_len = cache[0].offset if cache is not None and cache[0] is not None else 0
    total_seq_len = history_len + query_seq_len

    # Create mask allowing each query position to only attend up to its position
    # Shape will be [query_seq_len, total_seq_len]
    mask = create_causal_mask(query_seq_len=query_seq_len, total_seq_len=total_seq_len)
    return mask.astype(h.dtype)


def create_causal_mask(query_seq_len: int, total_seq_len: int) -> mx.array:
    """Creates a causal mask for attention, explicitly handling the two-token case.

    Args:
        query_seq_len: Number of new query positions (typically 2 for two-token decoding)
        total_seq_len: Total sequence length including history (history_len + query_seq_len)

    Returns:
        Mask tensor of shape [query_seq_len, total_seq_len] where -inf (as -1e9)
        blocks attending to future positions

    Example for query_seq_len=2, total_seq_len=4:
        [[-inf, -inf,    0,    0],  # First query can attend to positions 2,3
         [-inf, -inf, -inf,    0]]  # Second query can only attend to position 3
    """
    # Create position indices
    query_positions = mx.arange(query_seq_len)  # [0, 1]
    key_positions = mx.arange(total_seq_len)  # [0, 1, 2, 3]

    # Calculate valid attention positions
    # Each query can attend up to: history_len + its_position
    history_len = total_seq_len - query_seq_len
    valid_attention = (query_positions[:, None] + history_len) >= key_positions[None, :]

    # Convert to attention mask (-inf for invalid positions)
    mask = mx.where(valid_attention, 0, -1e9)
    return mask
