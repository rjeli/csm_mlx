from dataclasses import dataclass
from huggingface_hub import snapshot_download
from pathlib import Path
import mlx.core as mx
import numpy as np
import time
from tokenizers import Tokenizer
from typing import Optional, List, Tuple
from tqdm import tqdm

from csm_mlx.codec.mimi import load_mimi, MimiModel
from csm_mlx.lm.rq_transformer import (
    RQTransformerModelArgs,
)
from csm_mlx.lm.csm import CSMModel
from csm_mlx.lm.cache import make_prompt_cache
from csm_mlx.lm.config import ModelType
from csm_mlx.lm.utils.prompt import CSMPromptEncoder
from csm_mlx.generate.csm import SingleBatchGenerator
from csm_mlx.generate.utils import GenerationSettings
from csm_mlx.io.wav import pcm_to_wav_bytes


@dataclass
class Segment:
    speaker: int
    text: str
    audio: np.ndarray
    """
    24khz pcm ndarray
    """


class CSM:
    model: CSMModel
    config: RQTransformerModelArgs
    codec: MimiModel
    prompt_encoder: CSMPromptEncoder

    def __init__(
        self, model_id="jkeisling/csm-1b", checkpoint_dir: Optional[str] = None
    ):
        checkpoint_dir = Path(
            checkpoint_dir
            if checkpoint_dir is not None
            else snapshot_download(model_id)
        )
        config = RQTransformerModelArgs.from_json_file(
            str(checkpoint_dir / "config.json")
        )
        model_type = ModelType.csm_1b()

        config = RQTransformerModelArgs.from_json_file(
            str(checkpoint_dir / "config.json")
        )
        tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
        prompt_encoder = CSMPromptEncoder(tokenizer)

        model = CSMModel(config, model_type)
        model_path = str(checkpoint_dir / "model.safetensors")
        model.load_weights(model_path, strict=True)
        model = model.apply(lambda p: p.astype(mx.bfloat16))
        mx.eval(model.parameters())
        model.eval()
        self.codec = load_mimi()
        self.model = model
        self.prompt_encoder = prompt_encoder
        self.sampling_rate = 24_000

    def __call__(
        self,
        text: str,
        speaker_id: int,
        context: Optional[List[Segment]] = None,
        temp: float = 0.9,
        fast_temp: float = 0.9,
        top_k: int = 64,
        backbone_min_p: float = 0.1,
    ) -> np.ndarray:
        """
        Blocking E2E audio generation; returns 24khz PCM
        """
        prompt, prompt_mask = self._prompt_encode(
            text, speaker_id, context if context is not None else []
        )

        decode_start_time = time.time()
        gen = SingleBatchGenerator(
            self.model,
            prompt,
            prompt_mask,
            GenerationSettings(
                default_temp=temp,
                default_fast_temp=fast_temp,
                top_k=top_k,
                min_p=backbone_min_p,
            ),
        )
        prefill_start_time = time.time()
        codes = [next(gen)]
        prefill_end_time = time.time()
        prefill_ms = (prefill_end_time - prefill_start_time) * 1000
        print(
            f"{prefill_ms:3f}ms prompt processing: {prompt.shape[1]} tokens ({prompt.shape[-1] / (prefill_end_time - prefill_start_time):3f} tokens/s)"
        )

        # accumulate codes blocking
        for step in tqdm(gen):
            if step is not None:
                codes.append(step)

        mx.eval(codes)

        out_len = len(codes) - 1
        codes = mx.concat(codes, axis=-1)
        decode_end_time = time.time()
        decode_duration = decode_end_time - decode_start_time
        frame_rate = 12.5
        pcm = self.codec.decode(codes)
        print(
            f"Generated in {decode_duration:.2f}s ({(out_len / decode_duration):.2f} tokens/s, {((decode_duration * 1000) / out_len):.2f}ms/token), {(out_len / frame_rate) / decode_duration:.2f}x realtime"
        )
        mx.metal.clear_cache()
        return np.array(pcm).flatten()

    def _prompt_encode(
        self, text: str, speaker_id: int, segments: List[Segment]
    ) -> Tuple[mx.array, mx.array]:
        """
        Returns text, text mask
        """
        prompt_segments = []
        mask_segments = []

        # TODO parallelize this if it ever gets slow
        for segment in segments:
            tokens, tokens_mask = self.prompt_encoder.tokenize_text(
                f"[{segment.speaker}]{segment.text}"
            )
            prompt_segments.append(tokens)
            mask_segments.append(tokens_mask)

            # TODO allow for precomputed prompt / cache; just getting it working for now
            # assumes audio is 24khz resampled elsewhere
            mimi_codes = self.codec.encode(
                mx.array(segment.audio)[mx.newaxis, mx.newaxis, :]
            )
            audio, audio_mask = self.prompt_encoder.tokenize_audio(mimi_codes)
            prompt_segments.append(audio)
            mask_segments.append(audio_mask)

        curr_tokens, curr_tokens_mask = self.prompt_encoder.tokenize_text(
            f"[{speaker_id}]{text}"
        )
        prompt_segments.append(curr_tokens)
        mask_segments.append(curr_tokens_mask)

        prompt = mx.concat(prompt_segments, axis=0)[mx.newaxis, :, :]
        prompt_mask = mx.concat(mask_segments, axis=0)[mx.newaxis, :, :]
        return prompt, prompt_mask
