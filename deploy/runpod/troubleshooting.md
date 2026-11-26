# RunPod troubleshooting

Problems hit while running vLLM on RunPod with either method in the [runbook](./README.md). Merged from the pip and image guides in [`reference/`](./reference/).

## Quick table

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: libcudart.so.13` | CUDA 13 vLLM wheel on a CUDA 12 pod (Method A) | [Install a CUDA 12 wheel](#importerror-libcudartso13-cannot-open-shared-object-file) |
| `no kernel image is available for execution on the device` | Image needs a newer driver than the pod has (Method B) | [Pick a CUDA 13.0 GPU or use Method A](#gpu-driver-compatibility-errors-method-b) |
| `Cannot connect to the Docker daemon` | `docker run` inside a pod | [Use Method B](./README.md#why-not-docker-run-inside-a-pod) |
| `502 Bad Gateway` from the proxy | vLLM still loading | Wait and retry `/health` |
| `404` model not found | Wrong `model` in the request | Use the id from `/v1/models` |
| `CUDA out of memory` at startup | `--gpu-memory-utilization` or `--max-model-len` too high | Lower to `0.85`, then lower `--max-model-len`, and restart |
| Port not reachable | Port 8000 not exposed on the pod | Add HTTP port 8000 in the pod settings |

## ImportError: libcudart.so.13: cannot open shared object file

Applies to Method A.

**Symptom.** `vllm serve` crashes immediately with a traceback ending in `ImportError: libcudart.so.13: cannot open shared object file`.

**Cause.** As of vLLM v0.21.0 the default wheel on PyPI targets CUDA 13 instead of CUDA 12. `uv pip install vllm` on a CUDA 12 pod pulls the CUDA 13 build, whose compiled extension (`vllm._C`) looks for `libcudart.so.13`, which CUDA 12 systems do not have.

**Check.**

```bash
nvidia-smi | head -4                                  # driver and CUDA version
nvcc --version                                        # toolkit, if installed
python -c "import torch; print(torch.version.cuda)"   # PyTorch's CUDA
uname -m                                              # architecture, e.g. x86_64
```

**Fix.** Find the CUDA 12 wheel for your architecture on a vLLM release page. [v0.20.2](https://github.com/vllm-project/vllm/releases/tag/v0.20.2) provides `cu129` wheels (for example `vllm-0.x.y+cu129-cp312-cp312-linux_x86_64.whl`). Then rebuild the environment:

```bash
deactivate
rm -rf /workspace/vllm-env
uv venv --python 3.12 --seed /workspace/vllm-env
source /workspace/vllm-env/bin/activate
uv pip install <WHEEL_URL>
python -c "import vllm; print(vllm.__version__)"
vllm --help
```

If PyTorch in the environment still does not match the pod's CUDA, reinstall it for CUDA 12.8:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"   # expect: True 12.8
```

TODO(owner): record whether the PyTorch reinstall was needed on top of the CUDA 12 wheel, or only one of the two fixes.

**Rule of thumb.**

| Pod CUDA | Install |
|---|---|
| 12.x | `--torch-backend=cu129`, or a CUDA 12 wheel directly |
| 13.x | `--torch-backend=auto` (selects cu130) |

**Seen on.** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` with vLLM installed after the v0.20.0 release (May 2026), and `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; the two reference documents name different images. TODO(owner): record which image the failure was actually reproduced on.

## GPU driver compatibility errors (Method B)

**Symptom.** The `vllm/vllm-openai` image fails at startup with:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

or

```text
The detected CUDA version mismatches the version that was used to compile PyTorch.
```

**Cause.** Recent vLLM images need CUDA 12.9 or newer (including 13.x) and a matching driver. Some RunPod GPUs ship older drivers.

**Constraint.** The driver is fixed by the RunPod host and cannot be updated from inside a pod.

**Fix.** Choose a GPU that reports CUDA 13.0 (observed: L40S, RTX PRO 4500, RTX PRO 4000; see the [compatibility table](./README.md#choose-a-method-gpu-and-cuda-compatibility)). If only a CUDA 12.x GPU is available (A40, L4, RTX 4090, RTX 5090), use Method A with a CUDA 12 wheel instead.
