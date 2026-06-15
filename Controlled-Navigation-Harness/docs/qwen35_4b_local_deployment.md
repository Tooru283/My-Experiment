# Qwen3.5-4B Local Deployment Plan

## Goal

Deploy `Qwen/Qwen3.5-4B` as a local OpenAI-compatible model service for Open-Nav.

The deployment must stay isolated from the existing `opennav` environment because Open-Nav depends on Habitat 0.1.7, Python 3.8, RAM, and SpatialBot. Modern Qwen3.5 serving stacks require Python 3.10+ and newer `transformers`, `pydantic`, and serving libraries.

## Model Capability

`Qwen/Qwen3.5-4B` supports image-text input. The Hugging Face repo is tagged as `image-text-to-text` and includes image/video preprocessor config files.

Two usage modes are planned:

1. Text-only Open-Nav replacement:
   - Use the model as a local replacement for GPT-style chat completion.
   - Run vLLM with `--language-model-only`.
   - Open-Nav continues using RAM + SpatialBot3B to convert visual observations into text.

2. Future text+image mixed input:
   - Run Qwen3.5-4B without `--language-model-only`.
   - Modify Open-Nav to send multimodal OpenAI-compatible messages containing text plus image payloads.
   - This is a larger code change because the current Open-Nav navigator sends text-only prompts.

## Environment Layout

Keep two conda environments:

- `opennav`
  - Python 3.8
  - Habitat/Open-Nav/RAM/SpatialBot
  - Runs `run.py` and experiment logic

- `qwen35-serve`
  - Python 3.11
  - vLLM/SGLang-style model server dependencies
  - Runs the local OpenAI-compatible API server

The experiment process calls the model server over HTTP:

```text
Open-Nav process -> http://127.0.0.1:23333/v1 -> Qwen3.5-4B service
```

## Paths

Model local path:

```text
/root/models/Qwen3.5-4B
```

Open-Nav project:

```text
/root/wjj/Open-Nav
```

## Installation Commands

Create the independent serving environment:

```bash
conda create -n qwen35-serve python=3.11 -y
conda activate qwen35-serve
python -m pip install -U pip uv
```

Install vLLM in the serving environment.

Qwen's README recommends a fresh environment and a vLLM nightly wheel for Qwen3.5. Use CUDA wheels for future GPU execution:

```bash
uv pip install vllm --torch-backend=cu128 --extra-index-url https://wheels.vllm.ai/nightly
```

If `cu128` is not available on the target GPU host, use:

```bash
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

## Model Download

Download model files to a fixed local directory:

```bash
hf download Qwen/Qwen3.5-4B --local-dir /root/models/Qwen3.5-4B --max-workers 4
```

Expected major files:

```text
model.safetensors-00001-of-00002.safetensors
model.safetensors-00002-of-00002.safetensors
model.safetensors.index.json
config.json
tokenizer.json
preprocessor_config.json
video_preprocessor_config.json
```

## Text-Only Service Command

This is the recommended first deployment mode for current Open-Nav:

```bash
conda activate qwen35-serve

CUDA_VISIBLE_DEVICES=0 vllm serve /root/models/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 23333 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --served-model-name Qwen/Qwen3.5-4B
```

Use `--max-model-len 8192` first. The official 262144 context example is too aggressive for small single-GPU deployments because KV cache memory dominates.

## Future Multimodal Service Command

When Open-Nav is changed to send image-text mixed messages, remove `--language-model-only`:

```bash
conda activate qwen35-serve

CUDA_VISIBLE_DEVICES=0 vllm serve /root/models/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 23333 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --served-model-name Qwen/Qwen3.5-4B
```

## Open-Nav Call Pattern

Open-Nav should keep running from the existing environment:

```bash
conda activate opennav
cd /root/wjj/Open-Nav
bash run_OpenNav.bash
```

The OpenAI client in Open-Nav should point to:

```python
OpenAI(
    api_key="not-needed",
    base_url="http://127.0.0.1:23333/v1",
)
```

Use model name:

```text
Qwen/Qwen3.5-4B
```

## Health Check

After the server starts, test it from any environment that has the `openai` Python package:

```bash
python - <<'PY'
from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:23333/v1")
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    temperature=0,
)
print(resp.choices[0].message.content)
PY
```

Expected output:

```text
ok
```

## Current Machine Constraint

The current machine has no visible GPU and only about 2 GiB RAM, so the service cannot be meaningfully started here. The executable checks on this machine are:

- create the isolated environment
- install serving dependencies
- download model files
- validate model config/tokenizer/preprocessor files
- confirm `vllm` command exists

Full model serving should be started after the GPU machine is attached.

## Execution Status on 2026-05-27

Completed on the current server:

- Created conda environment:

```text
/root/anaconda3/envs/qwen35-serve
```

- Installed serving stack in `qwen35-serve`:

```text
Python 3.11.15
torch 2.11.0+cu128
CUDA build 12.8
vLLM 0.21.1rc1.dev315+g0b68f21e7
transformers 5.9.0
openai 2.38.0
```

- Confirmed `opennav` was not changed:

```text
Python 3.8.20
torch 2.4.1+cu121
transformers 4.44.0
pydantic 2.10.6
```

- Downloaded `Qwen/Qwen3.5-4B` to:

```text
/root/models/Qwen3.5-4B
```

- Verified core model files:

```text
model.safetensors-00001-of-00002.safetensors 5329398688
model.safetensors-00002-of-00002.safetensors 3990429408
model.safetensors.index.json 76196
config.json 3161
tokenizer.json 12807982
preprocessor_config.json 390
video_preprocessor_config.json 385
```

- Verified model metadata:

```text
config Qwen3_5Config qwen3_5
tokenizer Qwen2Tokenizer 248077
processor Qwen3VLProcessor
```

- Verified safetensors shards can be opened:

```text
model.safetensors-00001-of-00002.safetensors: 87 tensors
model.safetensors-00002-of-00002.safetensors: 651 tensors
```

- Confirmed `vllm` entrypoint exists:

```text
/root/anaconda3/envs/qwen35-serve/bin/vllm
```

Current machine limitation:

```text
torch.cuda.is_available() == False
```

Because no GPU/NVML device is visible on this server, running `vllm --version` or starting `vllm serve` currently fails during device inference with:

```text
RuntimeError: Failed to infer device type
```

This is expected for the current no-GPU machine. The environment and model files are ready for GPU-side startup with the service command above.
