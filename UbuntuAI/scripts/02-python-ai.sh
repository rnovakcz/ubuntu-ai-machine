#!/bin/bash
#===============================================================================
# 02-python-ai.sh - Python + AI knihovny (kompatibilní verze)
# Ubuntu 25.10 AI Development Environment
#
# VERZE KOMPATIBILITY (prosinec 2024 - RTX 5060 Ti Blackwell):
#   Python: 3.11
#   PyTorch: 2.5.1 + CUDA 12.6 (Blackwell support)
#   TensorFlow: 2.18.0
#   JAX: 0.4.35
#===============================================================================

set -e
exec > >(tee -a /var/log/ai-setup.log) 2>&1

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok() { echo -e "${GREEN}[OK]${NC} $1"; }

[[ $EUID -ne 0 ]] && { echo "Spusťte jako root (sudo)"; exit 1; }
USER_REAL=${SUDO_USER:-$USER}
HOME_REAL=$(getent passwd "$USER_REAL" | cut -d: -f6)

log "========== PYTHON + AI KNIHOVNY =========="

#--- Mambaforge ---
log "Instalace Mambaforge..."
MAMBA_DIR="$HOME_REAL/mambaforge"
if [ ! -d "$MAMBA_DIR" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O /tmp/mamba.sh
    sudo -u "$USER_REAL" bash /tmp/mamba.sh -b -p "$MAMBA_DIR"
    rm /tmp/mamba.sh
    sudo -u "$USER_REAL" "$MAMBA_DIR/bin/conda" init bash
fi

#--- Hlavní AI prostředí ---
log "Vytváření AI prostředí s kompatibilními verzemi..."
sudo -u "$USER_REAL" bash << 'AIENV'
source ~/mambaforge/etc/profile.d/conda.sh
source ~/mambaforge/etc/profile.d/mamba.sh

mamba create -n ai python=3.11 -y
conda activate ai

#===============================================================================
# CORE ML FRAMEWORKS (kompatibilní verze)
#===============================================================================

# PyTorch 2.5.1 + CUDA 12.6 (pro RTX 5060 Ti Blackwell)
# Poznámka: cu126 wheel může být nightly, pokud není dostupný, použije se cu124
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu126 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# TensorFlow 2.18 (kompatibilní s CUDA 12.x)
pip install tensorflow==2.18.0 tensorflow-hub==0.16.1 tensorflow-datasets==4.9.6

# JAX 0.4.35 s CUDA 12
pip install "jax[cuda12]==0.4.35"

#===============================================================================
# SCIENTIFIC STACK
#===============================================================================
mamba install -y \
    numpy=1.26.4 \
    scipy=1.14.1 \
    pandas=2.2.3 \
    matplotlib=3.9.2 \
    seaborn=0.13.2 \
    scikit-learn=1.5.2 \
    jupyter=1.1.1 \
    jupyterlab=4.2.5 \
    ipywidgets=8.1.5

#===============================================================================
# HUGGING FACE ECOSYSTEM
#===============================================================================
pip install \
    transformers==4.46.3 \
    datasets==3.1.0 \
    tokenizers==0.20.3 \
    accelerate==1.1.1 \
    peft==0.13.2 \
    trl==0.12.1 \
    bitsandbytes==0.44.1 \
    safetensors==0.4.5 \
    sentencepiece==0.2.0 \
    diffusers==0.31.0 \
    huggingface-hub==0.26.2

#===============================================================================
# IBM WATSONX + GRANITE
#===============================================================================
pip install \
    ibm-watsonx-ai==1.1.16 \
    ibm-watson-machine-learning==1.0.360 \
    ibm-generative-ai==3.0.0

#===============================================================================
# LANGCHAIN + AGENTS
#===============================================================================
pip install \
    langchain==0.3.7 \
    langchain-community==0.3.7 \
    langchain-core==0.3.19 \
    langchain-huggingface==0.1.2 \
    langgraph==0.2.53 \
    llama-index==0.11.22

#===============================================================================
# VECTOR DATABASES
#===============================================================================
pip install \
    chromadb==0.5.18 \
    faiss-cpu==1.9.0 \
    qdrant-client==1.12.1 \
    pinecone-client==5.0.1

#===============================================================================
# MODEL SERVING
#===============================================================================
pip install \
    vllm==0.6.4 \
    gradio==5.6.0 \
    streamlit==1.40.1 \
    fastapi==0.115.5 \
    uvicorn==0.32.1

#===============================================================================
# COMPUTER VISION
#===============================================================================
pip install \
    opencv-python==4.10.0.84 \
    ultralytics==8.3.40 \
    supervision==0.25.0 \
    albumentations==1.4.21 \
    timm==1.0.11

#===============================================================================
# AUDIO/SPEECH
#===============================================================================
pip install \
    openai-whisper==20240930 \
    faster-whisper==1.0.3 \
    librosa==0.10.2.post1 \
    soundfile==0.12.1

#===============================================================================
# NLP
#===============================================================================
pip install spacy==3.8.2 nltk==3.9.1 gensim==4.3.3

#===============================================================================
# MLOPS
#===============================================================================
pip install \
    mlflow==2.18.0 \
    wandb==0.18.7 \
    optuna==4.1.0 \
    tensorboard==2.18.0

#===============================================================================
# PYTORCH ECOSYSTEM
#===============================================================================
pip install \
    pytorch-lightning==2.4.0 \
    torchmetrics==1.5.2

#===============================================================================
# NVIDIA TOOLS
#===============================================================================
pip install \
    nvidia-ml-py==12.560.30 \
    pynvml==11.5.3 \
    tritonclient[all]==2.51.0

#===============================================================================
# EXPLAINABILITY
#===============================================================================
pip install shap==0.46.0 lime==0.2.0.1 captum==0.7.0

#===============================================================================
# INFERENCE OPTIMIZATION
#===============================================================================
pip install \
    onnx==1.17.0 \
    onnxruntime-gpu==1.20.1 \
    auto-gptq==0.7.1

#===============================================================================
# UTILITIES
#===============================================================================
pip install httpie rich tqdm python-dotenv pyyaml requests aiohttp

# Jupyter kernel
python -m ipykernel install --user --name ai --display-name "Python (AI)"

# SpaCy modely
python -m spacy download en_core_web_sm || true
python -m spacy download en_core_web_lg || true
AIENV

#--- Poetry ---
log "Instalace Poetry..."
sudo -u "$USER_REAL" bash -c 'curl -sSL https://install.python-poetry.org | python3 -'

#--- pipx nástroje ---
log "Instalace CLI nástrojů..."
apt install -y pipx
sudo -u "$USER_REAL" pipx ensurepath
sudo -u "$USER_REAL" pipx install black ruff pre-commit cookiecutter

#--- JupyterLab service ---
log "Vytváření JupyterLab service..."
cat > /etc/systemd/system/jupyterlab.service << EOF
[Unit]
Description=JupyterLab
After=network.target

[Service]
Type=simple
User=$USER_REAL
WorkingDirectory=$HOME_REAL/ai-workspace
Environment="PATH=$HOME_REAL/mambaforge/envs/ai/bin:/usr/local/bin:/usr/bin"
ExecStart=$HOME_REAL/mambaforge/envs/ai/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
Restart=always

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload

#--- Test skript ---
cat > "$HOME_REAL/ai-workspace/scripts/test-gpu.py" << 'EOF'
#!/usr/bin/env python3
"""Test GPU a verzí AI knihoven."""
print("="*60)
print("AI ENVIRONMENT TEST")
print("="*60)

import sys
print(f"\nPython: {sys.version}")

try:
    import torch
    print(f"\n[PyTorch {torch.__version__}]")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
except Exception as e:
    print(f"PyTorch: {e}")

try:
    import tensorflow as tf
    print(f"\n[TensorFlow {tf.__version__}]")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPUs: {len(gpus)}")
except Exception as e:
    print(f"TensorFlow: {e}")

try:
    import jax
    print(f"\n[JAX {jax.__version__}]")
    print(f"  Devices: {jax.devices()}")
except Exception as e:
    print(f"JAX: {e}")

try:
    import transformers
    print(f"\n[Transformers {transformers.__version__}]")
except Exception as e:
    print(f"Transformers: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
EOF
chmod +x "$HOME_REAL/ai-workspace/scripts/test-gpu.py"
chown -R "$USER_REAL:$USER_REAL" "$HOME_REAL/ai-workspace"

ok "========== PYTHON + AI HOTOVO =========="
log "Aktivace: conda activate ai"
log "Test: python ~/ai-workspace/scripts/test-gpu.py"
