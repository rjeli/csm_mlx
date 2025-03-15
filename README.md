# Sesame CSM 1B MLX port

Port of Sesame's [CSM](https://github.com/SesameAILabs/csm) model to [MLX](https://github.com/ml-explore/mlx) for use on Apple Silicon.

The project goal is realtime streaming inference on a MacBook.

## Installation

Clone this repo.

Get [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already. Then:

```
uv sync
```

Then open `example.ipynb` for basic inference!

## Roadmap

- [x] Safetensors conversion
- [x] Core modeling and entry point
- [ ] Streaming output
- [ ] Gradio UI
- [ ] FastRTC speech-to-speech webui
- [ ] PyPI library
