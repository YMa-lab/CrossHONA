FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    WANDB_MODE=offline \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app

# Build env first so we can cache it across code changes
COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml && conda clean -afy

# umap-learn is optional but nicer than PCA fallback
RUN /opt/conda/envs/crossspecies/bin/pip install --no-cache-dir umap-learn

# Pull in app code + the existing scripts
COPY . /app/
# scripts_hierarchical is mounted at runtime via docker-compose, so we don't bake it in

ENV CROSSSPECIES_DATA_DIR=/app/data \
    CROSSSPECIES_RESULTS_DIR=/app/results \
    CROSSSPECIES_SCRIPTS_DIR=/app/scripts_hierarchical \
    OLLAMA_BASE_URL=http://ollama:11434/v1 \
    OLLAMA_MODEL=llama3.1:8b

EXPOSE 7860

SHELL ["/bin/bash", "-lc"]
CMD ["bash", "-lc", "source activate crossspecies && python app.py"]
