#!/usr/bin/env -S uv run

import argparse
import os
import typing as t
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from numpy.typing import NDArray
from scipy.signal import resample
from tqdm import tqdm

from csm_mlx.generate.utils import GenerationSettings
from csm_mlx.io.wav import pcm_to_wav_bytes
from csm_mlx.lm.csm import CSMModel
from csm_mlx.lm.utils.samplers import min_p_sampling, top_k_sampling
from csm_mlx.loaders import CSM
from csm_mlx.loaders.csm import Segment


def load_wav(path: str) -> NDArray[np.float64]:
    wav, sr = sf.read(path)
    new_sample_rate = model.sampling_rate
    num_samples = int(len(wav) * new_sample_rate / sr)
    return t.cast(NDArray[np.float64], resample(wav, num_samples))


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if (
        kv_bits is not None
        and not isinstance(prompt_cache[0], QuantizedKVCache)
        and prompt_cache[0].offset > quantized_kv_start
    ):
        for i in range(len(prompt_cache)):
            if isinstance(prompt_cache[i], KVCache):
                prompt_cache[i] = prompt_cache[i].to_quantized(
                    group_size=kv_group_size, bits=kv_bits
                )


def generate_step(
    *,
    model: CSMModel,
    y: mx.array,
    y_mask: mx.array,
    backbone_sampler: t.Optional[t.Callable[[mx.array], mx.array]] = None,
    fast_sampler: t.Optional[t.Callable[[mx.array], mx.array]] = None,
    prefill_step_size: int = 32,
    max_tokens: int = 1024,
    kv_bits: int | None = None,
    kv_group_size: int | None = None,
    prompt_progress_callback: t.Optional[t.Callable[[int, int], None]] = None,
    stream: t.Optional[mx.Stream] = None,
) -> t.Generator[t.Any, None, None]:
    print(f"{y.shape=} {y_mask.shape=}")

    stream = stream or mx.new_stream(mx.default_device())

    backbone_sampler = backbone_sampler or (lambda x: mx.argmax(x, axis=-1))
    fast_sampler = fast_sampler or (lambda x: mx.argmax(x, axis=-1))
    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    if kv_group_size is None:
        kv_group_size = 32 if kv_bits == 4 else 64

    prompt_cache = [
        QuantizedKVCache(bits=kv_bits, group_size=kv_group_size)
        if kv_bits is not None
        else KVCache()
        for _ in range(len(model.layers))
    ]

    quantize_cache_fn = partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def fast_decode(y: mx.array, h: mx.array):
        yy = mx.concat([y, mx.zeros(32, dtype=mx.int64)])
        decoder_cache = [QuantizedKVCache() for _ in range(len(model.fast_layers))]

        max_codebooks = model.config.num_codebooks
        actual_codebooks = max_codebooks - 0

        for i in range(1, actual_codebooks):
            ci_logits = model.forward_generate_fast(
                x=h[None],
                cache=decoder_cache,
                codebook_idx=i,
            )
            ci = fast_sampler(ci_logits)
            yy[i] = ci
            h = model.embed_audio(i, ci)
        y = yy[None]

        y_mask = mx.zeros_like(y)
        y_mask[:, :actual_codebooks] = 1

        return y, y_mask

    def _step(y, y_mask):
        c0_logits, h = model.forward_generate(
            y[None],
            y_mask[None],
            prompt_cache,
        )
        y = backbone_sampler(c0_logits)
        y_emb = model.embed_audio(0, y)
        h = mx.concat([h, y_emb], axis=0)
        return fast_decode(y, h)

    with mx.stream(stream):
        prompt_processed_tokens = 0
        total_prompt_tokens = y.shape[0]
        while y.shape[0] > prefill_step_size:
            _ = model.forward_generate(
                y[:prefill_step_size][None],
                y_mask[:prefill_step_size][None],
                prompt_cache,
            )
            mx.eval([c.state for c in prompt_cache])
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt_processed_tokens += prefill_step_size
            y = y[prefill_step_size:]
            y_mask = y_mask[prefill_step_size:]
            mx.metal.clear_cache()
        prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        y, y_mask = _step(y, y_mask)
        mx.eval(y, y_mask)

    # hmm
    """
    _step = mx.compile(
        _step,
        inputs=[mx.random.state, prompt_cache],
        outputs=[mx.random.state, prompt_cache],
    )
    """

    for n in range(max_tokens):
        with mx.stream(stream):
            next_y, next_y_mask = _step(y, y_mask)
        mx.async_eval(next_y, next_y_mask)
        yield y[0, :-1]
        if n % 256 == 0:
            mx.metal.clear_cache()
        y, y_mask = next_y, next_y_mask


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("-q", type=int, choices=[4, 8])
    argp.add_argument("-qkv", type=int, choices=[4, 8])
    args = argp.parse_args()

    print(args)

    model = CSM()
    print(mx.metal.device_info())
    print(f"{mx.default_device()=}")

    # mx.metal.set_wired_limit(10 * (1024**3))

    # possibly faster but not for me
    # model.model.set_dtype(mx.float16)

    if args.q is not None:
        nn.quantize(
            model.model,
            bits=args.q,
            group_size=32 if args.q == 4 else 64,
        )

    import inspect

    text = inspect.cleandoc(
        """
        Its as if you have a girl you desire, she dies, but using big magic
        you reanimate her corpse, put makeup on her, reteach this zombie to speak,
        force her to copy all of her old habits, condition her like you
        would a pigeon to act in ways you remember and that you
        liked. But in the end shes just a reanimated live-action doll,
        and this is grotesque.
        """
        if True
        else """
        A mirror and exaltation of the false intellect of the nerd, that 
        never leaves the stream of words, syllogisms, motives and desire,
        that is always forced and contrived, because its under pressure of
        some petty need. And its really grotesque.
        """
    ).replace("\n", " ")
    print(f"{text=}")

    speaker_id = 0
    context = [
        Segment(
            speaker=0,
            text="When I heard the release demo, I was shocked, angered, and in disbelief that Mr. Altman would pursue a voice that sounded so eerily similar to mine that my closest friends and news outlets could not tell the difference.",
            audio=load_wav("./tests/sky.wav"),
        )
    ]

    gen_settings = GenerationSettings(
        default_temp=0.9,
        default_fast_temp=0.9,
        min_p=0.1,
        top_k=64,
        max_new_tokens=256,
    )

    prompt, prompt_mask = model._prompt_encode(text, speaker_id, context)
    prompt, prompt_mask = prompt[0], prompt_mask[0]
    mx.eval(prompt, prompt_mask)

    print(f"{prompt.dtype=} {prompt.shape=} {prompt_mask.dtype=} {prompt_mask.shape=}")

    gen = generate_step(
        model=model.model,
        y=prompt,
        y_mask=prompt_mask,
        backbone_sampler=partial(min_p_sampling, temperature=0.9, min_p=0.1),
        fast_sampler=partial(top_k_sampling, temperature=0.9, top_k=64),
        kv_bits=args.qkv,
        max_tokens=256,
        prompt_progress_callback=lambda i, tot: print(f"prompt: {i}/{tot}"),
    )

    codes = []

    for i, code in enumerate(tqdm(gen)):
        if "MTL_CAPTURE_ENABLED" in os.environ:
            if i == 5:
                print("start cap")
                mx.metal.start_capture("trace.gputrace")
            if i == 6:
                print("stop cap")
                mx.metal.stop_capture()
        if mx.all(code == 0):
            break
        # print(code.tolist())
        codes.append(code)

    codes = mx.stack(codes).T[None]
    print(f"{codes.shape=}")
    pcm = model.codec.decode(codes)
    pcm = np.array(pcm).flatten()
    print(pcm.shape)

    q = "q" + str(args.q) if args.q is not None else "bf16"
    # with open(f"out_{q}.wav", "wb") as f:
    with open("out.wav", "wb") as f:
        f.write(pcm_to_wav_bytes(pcm))
