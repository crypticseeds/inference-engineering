# Troubleshooting: vLLM Docker on RunPod

## GPU Driver Compatibility Errors

**Symptom:** When attempting to run the vLLM Docker image on RunPod, you may encounter GPU driver compatibility errors similar to:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

or

```
The detected CUDA version mismatches the version that was used to compile PyTorch.
```

**Root cause:** The latest vLLM Docker images require **CUDA 12.9 or higher** (including CUDA 13.x) and a corresponding modern GPU driver. Some RunPod GPU offerings ship with older driver versions that do not meet this requirement.

**Important constraint:** GPU drivers **cannot** be updated from within a RunPod pod. The driver version is fixed by the host machine and cannot be changed at the container level.

---

## RunPod GPU Compatibility Reference

The table below lists observed GPU driver and CUDA versions on RunPod. Use this to quickly identify which GPUs are compatible with the latest vLLM Docker images.

> **Note:** This is not an exhaustive list. Driver versions may vary between RunPod Secure Cloud and Community Cloud, and are subject to change as RunPod updates their infrastructure.

| GPU | Driver Version | CUDA Version | vLLM Docker Compatible |
|-----|---------------|--------------|------------------------|
| NVIDIA L40S | 580.126.09 | 13.0 | Yes |
| NVIDIA RTX PRO 4500 | 580.126.09 | 13.0 | Yes |
| NVIDIA RTX PRO 4000 | 580.159.04 | 13.0 | Yes |
| NVIDIA A40 | 570.211.01 | 12.8 | No |
| NVIDIA L4 | 570.195.03 | 12.8 | No |
| NVIDIA GeForce RTX 5090 | 570.195.03 | 12.8 | No |
| NVIDIA GeForce RTX 4090 | 565.57.01 | 12.7 | No |

### Recommendation

**Use a CUDA 13.0 GPU** (L40S, RTX PRO 4500, or RTX PRO 4000) when running vLLM via Docker on RunPod. These are also a good budget-conscious choice — they are often available at competitive rates despite their newer driver support.

---

## Workaround for CUDA 12.x GPUs

If you are on a budget and can only access a CUDA 12.x GPU (A40, L4, RTX 4090, RTX 5090), you can still run vLLM — just not via Docker. Install vLLM directly using `pip` or `uv` instead, which gives you more control over the CUDA and PyTorch versions.

See the [pip/uv installation and troubleshooting guide](../01-vllm-pip/installation.md) for step-by-step instructions.
