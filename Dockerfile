FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Official VICReg code provides the ResNet definition used by the self-supervised backbone.
RUN git clone --depth 1 https://github.com/facebookresearch/vicreg.git /app/vicreg

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY scripts ./scripts
COPY configs ./configs

CMD ["/bin/bash"]
