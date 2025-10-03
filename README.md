### Introduction

Our aim is to serve `Qwen/Qwen3-Embedding-0.6B` and myriad other embedding models using Ray Serve, specificially, we're aiming to follow the [Multi-application design pattern for Ray Serve.](https://docs.ray.io/en/latest/serve/multi-app.html)

### How to setup your environment for testing & development
**OPTIONAL (if you're CUDA drivers aren't updated, etc.):** On VSCode, set up the `devcontainer.json` by clicking `CTRL` + `SHIFT` + `p` > Reopen in Container.

Set up `uv`. Really goated package manager. It's blazing fast! Other methods [here]((https://docs.astral.sh/uv/getting-started/installation/#installation-methods)).

```bash
pip install --upgrade pip \
pip install uv \
# uv self update
```

Once everything is set up run the below:

```bash
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml --group dev # to add dev dependencies
```

### How to deploy the project (Locally)

```bash
serve build app.text_embedding:app -o config.yaml # generate `config.yaml` (if you haven't)
ray start --head --dashboard-port=8265
serve run config.yaml

ray stop # shut down ray cluster once your done testing
```

### How to deploy the project (with Docker)
I can't get the dashboard (at port 8265) up for some reason!

```
docker build -t ray-embedding-service .
docker run -it --rm --gpus all -p 8000:8000 -p 8265:8265 -p 6379:6379 ray-embedding-service
```

### To do's
[] Serve colpali model
[] Add dynamic check to see if ray cluster is up in `scripts/entrypoint.sh`

### Resources

- Multi-application for Ray Serve (main project inspiration): https://docs.ray.io/en/latest/serve/multi-app.html
- For FastAPI integration: https://github.com/ray-project/ray/blob/cfcc68f13798eb5c2c9888a089d4b9c95d21b7fc/python/ray/serve/tests/test_fastapi.py#L153-L325
- How to install `flash-attn` with `--no-build-isolation` using `uv`: https://github.com/astral-sh/uv/issues/6437#issuecomment-3167274955
- https://stackoverflow.com/questions/67468439/vs-code-devcontainers-what-is-the-difference-between-remoteuser-and-containeru