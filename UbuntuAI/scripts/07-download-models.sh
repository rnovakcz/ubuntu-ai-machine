#!/bin/bash
#===============================================================================
# 07-download-models.sh - Stažení lokálních modelů (bez placených účtů)
# Ubuntu 25.10 AI Development Environment
#===============================================================================

set -e

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

USER_REAL=${SUDO_USER:-$USER}
HOME_REAL=$(getent passwd "$USER_REAL" | cut -d: -f6)
MODELS_DIR="$HOME_REAL/ai-workspace/models"

log "========== STAHOVÁNÍ LOKÁLNÍCH MODELŮ =========="
warn "Toto stáhne několik GB dat!"

mkdir -p "$MODELS_DIR"/{ollama,huggingface,embeddings}

#===============================================================================
# OLLAMA MODELY (vyžaduje běžící ollama)
#===============================================================================
log "Stahování Ollama modelů..."

if command -v ollama &> /dev/null; then
    # Základní LLM modely
    ollama pull llama3.2          # 2GB - všeobecný, rychlý
    ollama pull mistral           # 4GB - kvalitní, rychlý
    ollama pull phi3              # 2GB - Microsoft, dobrý na kód
    ollama pull gemma2:2b         # 1.5GB - Google, malý a rychlý
    
    # Kódování
    ollama pull codellama         # 4GB - Meta, specializovaný na kód
    ollama pull deepseek-coder:6.7b  # 4GB - DeepSeek, výborný na kód
    
    # Embeddings (pro RAG)
    ollama pull nomic-embed-text  # 275MB - embeddings
    ollama pull mxbai-embed-large # 670MB - lepší embeddings
    
    # Vision (multimodal)
    ollama pull llava:7b          # 4.5GB - vision + text
    
    ollama list
    ok "Ollama modely staženy"
else
    warn "Ollama není nainstalována, přeskakuji"
fi

#===============================================================================
# HUGGING FACE MODELY (Python)
#===============================================================================
log "Stahování Hugging Face modelů..."

if [ -d "$HOME_REAL/mambaforge" ]; then
    sudo -u "$USER_REAL" bash << 'HFMODELS'
source ~/mambaforge/etc/profile.d/conda.sh
conda activate ai

python << 'PYTHON'
from huggingface_hub import snapshot_download
import os

cache_dir = os.path.expanduser("~/ai-workspace/models/huggingface")
os.makedirs(cache_dir, exist_ok=True)

models = [
    # Text classification
    "distilbert-base-uncased-finetuned-sst-2-english",
    
    # Text generation (malé)
    "distilgpt2",
    "microsoft/phi-2",
    
    # Question answering
    "distilbert-base-cased-distilled-squad",
    
    # NER
    "dslim/bert-base-NER",
    
    # Sentence embeddings
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    
    # Fill mask
    "distilbert-base-uncased",
    
    # Translation (malé)
    "Helsinki-NLP/opus-mt-en-cs",
    "Helsinki-NLP/opus-mt-cs-en",
]

print("Stahování Hugging Face modelů...")
for model in models:
    try:
        print(f"  Downloading: {model}")
        snapshot_download(model, cache_dir=cache_dir)
    except Exception as e:
        print(f"  Error: {e}")

print("Hotovo!")
PYTHON
HFMODELS
    ok "Hugging Face modely staženy"
else
    warn "Mambaforge není nainstalován, přeskakuji HF modely"
fi

#===============================================================================
# WHISPER MODELY
#===============================================================================
log "Stahování Whisper modelů..."

if [ -d "$HOME_REAL/mambaforge" ]; then
    sudo -u "$USER_REAL" bash << 'WHISPER'
source ~/mambaforge/etc/profile.d/conda.sh
conda activate ai

python << 'PYTHON'
# Faster Whisper modely
try:
    from faster_whisper import WhisperModel
    print("Stahování Whisper tiny...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("Stahování Whisper base...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Whisper modely staženy!")
except ImportError:
    print("faster-whisper není nainstalován")
except Exception as e:
    print(f"Error: {e}")
PYTHON
WHISPER
    ok "Whisper modely staženy"
fi

#===============================================================================
# YOLO MODELY
#===============================================================================
log "Stahování YOLO modelů..."

if [ -d "$HOME_REAL/mambaforge" ]; then
    sudo -u "$USER_REAL" bash << 'YOLO'
source ~/mambaforge/etc/profile.d/conda.sh
conda activate ai

python << 'PYTHON'
from ultralytics import YOLO

models = ["yolov8n.pt", "yolov8s.pt"]  # nano a small

for m in models:
    print(f"Stahování {m}...")
    model = YOLO(m)

print("YOLO modely staženy!")
PYTHON
YOLO
    ok "YOLO modely staženy"
fi

#===============================================================================
# SPACY MODELY
#===============================================================================
log "Stahování SpaCy modelů..."

if [ -d "$HOME_REAL/mambaforge" ]; then
    sudo -u "$USER_REAL" bash << 'SPACY'
source ~/mambaforge/etc/profile.d/conda.sh
conda activate ai

python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
SPACY
    ok "SpaCy modely staženy"
fi

#===============================================================================
# PŘEHLED
#===============================================================================
ok "========== STAHOVÁNÍ DOKONČENO =========="
echo ""
log "Stažené modely:"
echo ""
echo "OLLAMA (pro chat a generování):"
echo "  llama3.2, mistral, phi3, gemma2:2b"
echo "  codellama, deepseek-coder"
echo "  nomic-embed-text (embeddings)"
echo "  llava:7b (vision)"
echo ""
echo "HUGGING FACE (transformers):"
echo "  distilbert, distilgpt2, phi-2"
echo "  sentence-transformers embeddings"
echo "  Helsinki-NLP překlady"
echo ""
echo "WHISPER (speech-to-text):"
echo "  tiny, base"
echo ""
echo "YOLO (detekce objektů):"
echo "  yolov8n, yolov8s"
echo ""
echo "SPACY (NLP):"
echo "  en_core_web_sm, en_core_web_md"
echo ""
log "Vše běží lokálně bez API klíčů! 🚀"

