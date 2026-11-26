# Deployment

Deployment runbooks for the same vLLM service across environments.

- [`modal/`](./modal/README.md): serverless A10, used for the published benchmark runs
- [`runpod/`](./runpod/README.md): GPU pods, with vLLM installed via pip or run from the official `vllm/vllm-openai` image
- `docker/` (planned): pinned container deployment
- `kubernetes/` (planned): Kubernetes deployment with queue-depth autoscaling
