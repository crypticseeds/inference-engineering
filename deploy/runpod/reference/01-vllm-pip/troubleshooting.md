# Troubleshooting

---

## ImportError: libcudart.so.13: cannot open shared object file

**Symptom**

Running `vllm serve` immediately crashes with a traceback ending in:

**What happened**

As of vLLM v0.21.0, the default wheel published to PyPI was switched from
CUDA 12 to CUDA 13. Running `uv pip install vllm` on a CUDA 12 host pulls the
CUDA 13 build. The compiled C extension (`vllm._C`) looks for
`libcudart.so.13`, which does not exist on CUDA 12 systems.

**How to check your CUDA version**

```bash
nvidia-smi | head -4      # shows driver + CUDA version
nvcc --version            # shows toolkit version if nvcc is installed
```

**Fix**

Get your machine architecture:
```bash
uname -m    # e.g. x86_64, aarch64
```

Find the matching CUDA 12 wheel on the [vLLM releases page](https://github.com/vllm-project/vllm/releases/tag/v0.20.2). Filter assets by your architecture and look for a filename containing `cu12` (e.g. `vllm-0.x.y+cu129-cp312-cp312-linux_x86_64.whl`).

for this demo, i am using "https://github.com/vllm-project/vllm/releases/tag/v0.20.2" which suports cu129

Copy the <WHEEL_URL>

Wipe the existing environment and reinstall, explicitly targeting the CUDA 12
wheel:

```bash
deactivate
rm -rf /workspace/vllm-env

uv venv --python 3.12 --seed /workspace/vllm-env
source /workspace/vllm-env/bin/activate

uv pip install <WHEEL_URL>
```

Verify before running anything else:

```bash
python -c "import vllm; print(vllm.__version__)"
vllm --help
```
# Activate the venv
source /workspace/vllm-env/bin/activate

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch for CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify torch sees the GPU
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"


**Rule of thumb going forward**

| System CUDA | Install flag |
|---|---|
| CUDA 12.x | `--torch-backend=cu129` |
| CUDA 13.x | `--torch-backend=auto` (picks cu130) |

Always check `nvidia-smi` before installing vLLM on a new machine.

**Affected environment**

RunPod base image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
with vLLM installed after v0.20.0 release (May 2026).
