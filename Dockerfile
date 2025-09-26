FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y curl git cmake\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY ./ .

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

RUN uv pip install --system --no-cache-dir -U pip

RUN uv pip install --system --no-cache-dir -r requirements.txt

RUN chmod -x launch_script.sh &&\
    sed -i 's/\r//g' launch_script.sh

ENTRYPOINT ["/bin/bash", "launch_script.sh"]
