#!/usr/bin/env -S uv run

import os
import time
import typing as t
from functools import partial

from numpy.typing import NDArray

from csm_mlx.lm.cache import KVCache, make_prompt_cache
from csm_mlx.lm.csm import CSMModel
from csm_mlx.lm.utils.samplers import min_p_sampling, top_k_sampling
from csm_mlx.loaders.csm import Segment

os.environ["MTL_CAPTURE_ENABLED"] = "1"

import mlx.core as mx
import numpy as np
import soundfile as sf
from einops import rearrange
from scipy.signal import resample
from tqdm import tqdm

from csm_mlx.generate.csm import SingleBatchGenerator
from csm_mlx.generate.utils import GenerationSettings
from csm_mlx.io.wav import pcm_to_wav_bytes
from csm_mlx.loaders import CSM


def load_wav(path: str) -> NDArray[np.float64]:
    wav, sr = sf.read(path)
    new_sample_rate = model.sampling_rate
    num_samples = int(len(wav) * new_sample_rate / sr)
    wav = resample(wav, num_samples)
    assert wav.dtype == np.float64
    return wav


model = CSM()

gen_settings = GenerationSettings(
    default_temp=0.9,
    default_fast_temp=0.9,
    min_p=0.1,
    top_k=64,
    max_new_tokens=1024,
)

text = "So, if you insist on this newscasting route, you're going to need to do some serious filtering with the poop"
speaker_id = 0
context = [
    Segment(
        speaker=0,
        text="When I heard the release demo, I was shocked, angered, and in disbelief that Mr. Altman would pursue a voice that sounded so eerily similar to mine that my closest friends and news outlets could not tell the difference.",
        audio=load_wav("./tests/sky.wav"),
    )
]

prompt, prompt_mask = model._prompt_encode(text, speaker_id, context)

gen = SingleBatchGenerator(model.model, prompt, prompt_mask, gen_settings)

codes = []

# accumulate codes blocking
for i, step in enumerate(tqdm(gen)):
    if step is not None:
        codes.append(step)

mx.eval(codes)

out_len = len(codes) - 1
codes = mx.concat(codes, axis=-1)

print(f"{codes.shape=}")

frame_rate = 12.5

pcm = model.codec.decode(codes)
# print(f"Generated in {decode_duration:.2f}s ({(out_len / decode_duration):.2f} tokens/s, {((decode_duration * 1000) / out_len):.2f}ms/token), {(out_len / frame_rate) / decode_duration:.2f}x realtime" )

mx.metal.clear_cache()
pcm = np.array(pcm).flatten()

print(pcm.shape)
with open("out.wav", "wb") as f:
    f.write(pcm_to_wav_bytes(pcm))
