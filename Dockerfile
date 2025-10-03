FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        poppler-utils \
    && wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip \
    && pip install --upgrade pip \
    && pip install uv \
    && apt-get purge -y --auto-remove software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN uv venv && . .venv/bin/activate && uv pip install -e .

RUN chmod +x /app/scripts/entrypoint.sh

EXPOSE 8000 8265 6379

CMD ["/app/scripts/entrypoint.sh"]
