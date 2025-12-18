# 🤖 Ubuntu AI Development Environment

Kompletní sada skriptů pro nastavení Ubuntu 25.10 jako AI vývojového prostředí.

## ✨ Co obsahuje

### Systém & Jazyky
- **NVIDIA**: Driver 565, CUDA 12.6, cuDNN 9, TensorRT, NCCL (RTX 5060 Ti ready!)
- **Python**: 3.11, Mambaforge, Conda/Mamba
- **Node.js**: 22 LTS, TypeScript 5.6, npm, yarn, pnpm
- **Go**: 1.23
- **Rust**: stable + cargo
- **Java**: OpenJDK 21, Maven, Gradle
- **.NET**: SDK 8, ML.NET, Semantic Kernel

### AI/ML Knihovny (kompatibilní verze)
- **PyTorch**: 2.5.1 + CUDA 12.4
- **TensorFlow**: 2.18
- **JAX**: 0.4.35
- **Hugging Face**: transformers, datasets, diffusers, PEFT, TRL
- **LangChain**: langchain, langgraph, llama-index
- **Computer Vision**: OpenCV, YOLO, supervision
- **Audio**: Whisper, faster-whisper, TTS
- **MLOps**: MLflow, Weights & Biases, Optuna

### Model Serving
- **Ollama**: Lokální LLM
- **Docker**: + NVIDIA Container Toolkit
- **Triton**: NVIDIA Inference Server
- **vLLM**: Rychlý LLM serving
- **Text Generation Inference**: Hugging Face

### GUI Nástroje
- **VS Code**: + 40+ rozšíření pro AI vývoj
- **JupyterLab**: jako systemd service
- **Postman, DBeaver, Obsidian**

## 🚀 Instalace

```bash
# Stáhněte nebo naklonujte
git clone <repo> && cd <repo>/scripts

# Udělejte spustitelným
chmod +x *.sh

# Spusťte hlavní instalátor
sudo ./install.sh
```

### Možnosti instalace

| Volba | Popis |
|-------|-------|
| `sudo ./install.sh` | Interaktivní menu |
| `sudo ./install.sh --all` | Kompletní instalace |
| `sudo ./install.sh --basic` | Bez modelů a příkladů |
| `sudo ./install.sh --models` | Pouze stažení modelů |

## 📁 Struktura po instalaci

```
~/
├── ai-workspace/           # Hlavní pracovní adresář
│   ├── models/             # Lokální modely
│   ├── datasets/           # Datasety
│   ├── projects/           # Vaše projekty
│   ├── notebooks/          # Jupyter notebooky
│   ├── scripts/            # Utility skripty
│   └── docker/             # Docker compose soubory
│
├── AI-Learning/            # Learning path
│   ├── 01-python-ai/       # NumPy, Pandas, vizualizace
│   ├── 02-pytorch-basics/  # Tensory, autograd, NN
│   ├── 03-transformers-llm/# HF, Ollama, embeddings
│   ├── 04-rag-system/      # RAG, ChromaDB
│   ├── 05-fine-tuning/     # LoRA, PEFT
│   ├── 06-computer-vision/ # CNN, YOLO
│   └── 07-audio-speech/    # Whisper, TTS
│
├── Examples/               # Ukázkové repozitáře
│   ├── nvidia/             # CUDA, TensorRT, Triton
│   ├── huggingface/        # Transformers, Diffusers
│   ├── ibm/                # Granite, watsonx
│   ├── pytorch/            # PyTorch examples
│   ├── langchain/          # LangChain, LlamaIndex
│   └── quick-start/        # Vlastní příklady
│
└── mambaforge/             # Conda environment
```

## 🎯 Lokální modely (bez API klíčů!)

### Ollama
```bash
ollama list                 # Seznam modelů
ollama pull llama3.2        # Stažení modelu
ollama run llama3.2         # Chat
```

### Hugging Face
```python
from transformers import pipeline
pipe = pipeline("sentiment-analysis")  # Automaticky stáhne model
```

### Stažené modely
- **LLM**: llama3.2, mistral, phi3, codellama
- **Embeddings**: nomic-embed-text, all-MiniLM-L6-v2
- **Vision**: llava:7b, yolov8n
- **Speech**: whisper tiny/base

## 🐳 Docker Stack

```bash
# Spuštění AI stacku
~/ai-workspace/scripts/start-ai-stack.sh

# Services:
# - Ollama:     http://localhost:11434
# - Open WebUI: http://localhost:3000
# - Qdrant:     http://localhost:6333
# - MinIO:      http://localhost:9001
```

## 📚 Learning Path

```bash
conda activate ai
cd ~/AI-Learning/01-python-ai
python numpy_basics.py
```

Doporučené pořadí:
1. `01-python-ai` - Základy dat
2. `02-pytorch-basics` - Deep learning
3. `03-transformers-llm` - Moderní NLP
4. `04-rag-system` - Praktické aplikace
5. `05-fine-tuning` - Customizace
6. `06-computer-vision` - Obrázky
7. `07-audio-speech` - Zvuk

## 🔧 Příkazy

```bash
# GPU test
nvidia-smi
python ~/ai-workspace/scripts/test-gpu.py

# Aktivace prostředí
conda activate ai

# JupyterLab
sudo systemctl start jupyterlab
# http://localhost:8888

# VS Code
code ~/ai-workspace
```

## 📋 Skripty

| Skript | Popis |
|--------|-------|
| `01-system-nvidia.sh` | Systém, NVIDIA, CUDA, jazyky |
| `02-python-ai.sh` | Python, Conda, AI knihovny |
| `03-docker-serving.sh` | Docker, Ollama, serving |
| `04-gui-vscode.sh` | VS Code, GUI nástroje |
| `05-examples.sh` | Ukázkové repozitáře |
| `06-learning-path.sh` | AI Learning Path |
| `07-download-models.sh` | Stažení modelů |
| `install.sh` | Hlavní instalátor |

## ⚠️ Požadavky

- Ubuntu 25.10 (nebo 24.04+)
- **NVIDIA GPU** - optimalizováno pro RTX 5060 Ti (Blackwell)
  - Driver: 565+
  - CUDA: 12.6
- 50GB+ volného místa
- 16GB+ RAM (32GB doporučeno)

## 🆘 Řešení problémů

```bash
# NVIDIA driver nefunguje
sudo ubuntu-drivers autoinstall
sudo reboot

# Conda nefunguje
source ~/.bashrc
# nebo
source ~/mambaforge/etc/profile.d/conda.sh

# Ollama neběží
sudo systemctl start ollama
ollama serve

# Docker permission denied
sudo usermod -aG docker $USER
# Odhlaste se a přihlaste
```

## 📄 Licence

MIT - volně použitelné pro osobní i komerční účely.

---

**Vše běží lokálně bez placených API účtů!** 🚀

