#!/bin/bash
#===============================================================================
# 05-examples.sh - Ukázkové repozitáře a příklady pro AI technologie
# Ubuntu 25.10 AI Development Environment
#===============================================================================

set -e
exec > >(tee -a /var/log/ai-setup.log) 2>&1

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok() { echo -e "${GREEN}[OK]${NC} $1"; }

[[ $EUID -ne 0 ]] && { echo "Spusťte jako root (sudo)"; exit 1; }
USER_REAL=${SUDO_USER:-$USER}
HOME_REAL=$(getent passwd "$USER_REAL" | cut -d: -f6)
EXAMPLES="$HOME_REAL/Examples"

log "========== UKÁZKOVÉ REPOZITÁŘE A PŘÍKLADY =========="

sudo -u "$USER_REAL" mkdir -p "$EXAMPLES"

#===============================================================================
# NVIDIA EXAMPLES
#===============================================================================
log "Stahování NVIDIA příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/nvidia"

# CUDA Samples
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git "$EXAMPLES/nvidia/cuda-samples" || true

# TensorRT Samples
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/NVIDIA/TensorRT.git "$EXAMPLES/nvidia/tensorrt" || true

# Triton Inference Server
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/triton-inference-server/server.git "$EXAMPLES/nvidia/triton-server" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/triton-inference-server/tutorials.git "$EXAMPLES/nvidia/triton-tutorials" || true

# NVIDIA NeMo (conversational AI)
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/NVIDIA/NeMo.git "$EXAMPLES/nvidia/nemo" || true

# NVIDIA Modulus (physics ML)
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/NVIDIA/modulus.git "$EXAMPLES/nvidia/modulus" || true

# RAPIDS (GPU data science)
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/rapidsai/notebooks.git "$EXAMPLES/nvidia/rapids-notebooks" || true

#===============================================================================
# HUGGING FACE EXAMPLES - KOMPLETNÍ
#===============================================================================
log "Stahování Hugging Face příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/huggingface"

# Core Libraries
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/transformers.git "$EXAMPLES/huggingface/transformers" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/datasets.git "$EXAMPLES/huggingface/datasets" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/tokenizers.git "$EXAMPLES/huggingface/tokenizers" || true

# Image Generation & Diffusion
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/diffusers.git "$EXAMPLES/huggingface/diffusers" || true

# Fine-tuning & Training
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/peft.git "$EXAMPLES/huggingface/peft" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/trl.git "$EXAMPLES/huggingface/trl" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/accelerate.git "$EXAMPLES/huggingface/accelerate" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/optimum.git "$EXAMPLES/huggingface/optimum" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/optimum-nvidia.git "$EXAMPLES/huggingface/optimum-nvidia" || true

# Inference & Serving
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/text-generation-inference.git "$EXAMPLES/huggingface/text-generation-inference" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/text-embeddings-inference.git "$EXAMPLES/huggingface/text-embeddings-inference" || true

# Multimodal
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/parler-tts.git "$EXAMPLES/huggingface/parler-tts" || true

# Agents & Tools
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/smolagents.git "$EXAMPLES/huggingface/smolagents" || true

# Evaluation
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/evaluate.git "$EXAMPLES/huggingface/evaluate" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/lighteval.git "$EXAMPLES/huggingface/lighteval" || true

# Alignment & Safety
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/alignment-handbook.git "$EXAMPLES/huggingface/alignment-handbook" || true

# Notebooks & Courses
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/notebooks.git "$EXAMPLES/huggingface/notebooks" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/course.git "$EXAMPLES/huggingface/course" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/diffusion-models-class.git "$EXAMPLES/huggingface/diffusion-course" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/deep-rl-class.git "$EXAMPLES/huggingface/deep-rl-course" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/audio-transformers-course.git "$EXAMPLES/huggingface/audio-course" || true

# Demos & Spaces
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/gradio-app/gradio.git "$EXAMPLES/huggingface/gradio" || true

# Specific Models & Tasks
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/setfit.git "$EXAMPLES/huggingface/setfit" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/autotrain-advanced.git "$EXAMPLES/huggingface/autotrain" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/safetensors.git "$EXAMPLES/huggingface/safetensors" || true

# LeRobot (robotics)
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/huggingface/lerobot.git "$EXAMPLES/huggingface/lerobot" || true

#===============================================================================
# IBM EXAMPLES
#===============================================================================
log "Stahování IBM příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/ibm"

# IBM Granite models
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/ibm-granite/granite-code-models.git "$EXAMPLES/ibm/granite-code" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/ibm-granite/granite-guardian.git "$EXAMPLES/ibm/granite-guardian" || true

# watsonx.ai samples
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/IBM/watson-machine-learning-samples.git "$EXAMPLES/ibm/watsonx-samples" || true

# IBM generative AI
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/IBM/ibm-generative-ai.git "$EXAMPLES/ibm/generative-ai" || true

#===============================================================================
# PYTORCH EXAMPLES
#===============================================================================
log "Stahování PyTorch příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/pytorch"

sudo -u "$USER_REAL" git clone --depth 1 https://github.com/pytorch/examples.git "$EXAMPLES/pytorch/examples" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/pytorch/tutorials.git "$EXAMPLES/pytorch/tutorials" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/Lightning-AI/pytorch-lightning.git "$EXAMPLES/pytorch/lightning" || true

#===============================================================================
# TENSORFLOW EXAMPLES
#===============================================================================
log "Stahování TensorFlow příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/tensorflow"

sudo -u "$USER_REAL" git clone --depth 1 https://github.com/tensorflow/examples.git "$EXAMPLES/tensorflow/examples" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/tensorflow/models.git "$EXAMPLES/tensorflow/models" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/keras-team/keras-io.git "$EXAMPLES/tensorflow/keras-examples" || true

#===============================================================================
# LANGCHAIN EXAMPLES
#===============================================================================
log "Stahování LangChain příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/langchain"

sudo -u "$USER_REAL" git clone --depth 1 https://github.com/langchain-ai/langchain.git "$EXAMPLES/langchain/langchain" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/langchain-ai/langgraph.git "$EXAMPLES/langchain/langgraph" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/run-llama/llama_index.git "$EXAMPLES/langchain/llama-index" || true

#===============================================================================
# COMPUTER VISION EXAMPLES
#===============================================================================
log "Stahování Computer Vision příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/computer-vision"

sudo -u "$USER_REAL" git clone --depth 1 https://github.com/ultralytics/ultralytics.git "$EXAMPLES/computer-vision/yolo" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/facebookresearch/detectron2.git "$EXAMPLES/computer-vision/detectron2" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/facebookresearch/segment-anything.git "$EXAMPLES/computer-vision/segment-anything" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/roboflow/supervision.git "$EXAMPLES/computer-vision/supervision" || true

#===============================================================================
# LLM & FINE-TUNING EXAMPLES
#===============================================================================
log "Stahování LLM příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/llm"

# Ollama
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/ollama/ollama.git "$EXAMPLES/llm/ollama" || true

# LitGPT (fine-tuning)
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/Lightning-AI/litgpt.git "$EXAMPLES/llm/litgpt" || true

# Axolotl (fine-tuning)
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/OpenAccess-AI-Collective/axolotl.git "$EXAMPLES/llm/axolotl" || true

# vLLM
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/vllm-project/vllm.git "$EXAMPLES/llm/vllm" || true

# LocalAI
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/mudler/LocalAI.git "$EXAMPLES/llm/localai" || true

#===============================================================================
# AUDIO/SPEECH EXAMPLES
#===============================================================================
log "Stahování Audio/Speech příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/audio"

sudo -u "$USER_REAL" git clone --depth 1 https://github.com/openai/whisper.git "$EXAMPLES/audio/whisper" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/coqui-ai/TTS.git "$EXAMPLES/audio/coqui-tts" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/suno-ai/bark.git "$EXAMPLES/audio/bark" || true

#===============================================================================
# DIGITAL TWIN & SIMULATION
#===============================================================================
log "Stahování Digital Twin příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/digital-twin"

# NVIDIA Omniverse
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/NVIDIA-Omniverse/kit-app-template.git "$EXAMPLES/digital-twin/omniverse-kit" || true

# Isaac Sim (robotics)
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git "$EXAMPLES/digital-twin/isaac-gym" || true

# PyBullet
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/bulletphysics/bullet3.git "$EXAMPLES/digital-twin/pybullet" || true

# MuJoCo examples
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/google-deepmind/mujoco.git "$EXAMPLES/digital-twin/mujoco" || true

#===============================================================================
# REINFORCEMENT LEARNING
#===============================================================================
log "Stahování RL příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/reinforcement-learning"

sudo -u "$USER_REAL" git clone --depth 1 https://github.com/DLR-RM/stable-baselines3.git "$EXAMPLES/reinforcement-learning/stable-baselines3" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/Farama-Foundation/Gymnasium.git "$EXAMPLES/reinforcement-learning/gymnasium" || true

#===============================================================================
# MLOPS EXAMPLES
#===============================================================================
log "Stahování MLOps příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/mlops"

sudo -u "$USER_REAL" git clone --depth 1 https://github.com/mlflow/mlflow.git "$EXAMPLES/mlops/mlflow" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/wandb/examples.git "$EXAMPLES/mlops/wandb-examples" || true
sudo -u "$USER_REAL" git clone --depth 1 https://github.com/iterative/dvc.git "$EXAMPLES/mlops/dvc" || true

#===============================================================================
# VLASTNÍ PŘÍKLADY - QUICK START NOTEBOOKY
#===============================================================================
log "Vytváření vlastních quick-start příkladů..."
sudo -u "$USER_REAL" mkdir -p "$EXAMPLES/quick-start"

# PyTorch Quick Start
cat > "$EXAMPLES/quick-start/01_pytorch_basics.py" << 'EOF'
#!/usr/bin/env python3
"""PyTorch Quick Start - Základy GPU."""
import torch

print(f"PyTorch verze: {torch.__version__}")
print(f"CUDA dostupná: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Jednoduchý tensor na GPU
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f"Matrix multiply na GPU: {z.shape}")
    
    # Jednoduché neuronové síť
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    ).cuda()
    
    input_data = torch.randn(32, 100, device='cuda')
    output = model(input_data)
    print(f"Model output: {output.shape}")
EOF

# Hugging Face Quick Start
cat > "$EXAMPLES/quick-start/02_huggingface_basics.py" << 'EOF'
#!/usr/bin/env python3
"""Hugging Face Quick Start - Transformers."""
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Sentiment analýza
print("=== Sentiment Analysis ===")
classifier = pipeline("sentiment-analysis")
result = classifier("I love using AI for development!")
print(result)

# Text generation
print("\n=== Text Generation ===")
generator = pipeline("text-generation", model="gpt2", device=0)
text = generator("AI is transforming", max_length=50, num_return_sequences=1)
print(text[0]['generated_text'])

# Vlastní model
print("\n=== Custom Model Loading ===")
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"Model loaded: {model_name}")
EOF

# LangChain Quick Start
cat > "$EXAMPLES/quick-start/03_langchain_basics.py" << 'EOF'
#!/usr/bin/env python3
"""LangChain Quick Start - RAG basics."""
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Vytvoření jednoduchého dokumentu
docs_content = """
Umělá inteligence (AI) je simulace lidské inteligence stroji.
Machine Learning je podmnožina AI zaměřená na učení z dat.
Deep Learning používá neuronové sítě s mnoha vrstvami.
LLM (Large Language Models) jsou velké jazykové modely trénované na textech.
"""

with open("/tmp/ai_docs.txt", "w") as f:
    f.write(docs_content)

# Načtení a rozdělení
loader = TextLoader("/tmp/ai_docs.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# Embedding a vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings)

# Dotaz
query = "Co je Machine Learning?"
docs = vectorstore.similarity_search(query, k=2)
print(f"Query: {query}")
print(f"Results: {[doc.page_content for doc in docs]}")
EOF

# NVIDIA CUDA Quick Start
cat > "$EXAMPLES/quick-start/04_cuda_basics.py" << 'EOF'
#!/usr/bin/env python3
"""NVIDIA CUDA Quick Start - GPU programování."""
import torch
import time

def benchmark_gpu():
    """Benchmark GPU vs CPU."""
    size = 10000
    
    # CPU
    cpu_start = time.time()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - cpu_start
    
    # GPU
    if torch.cuda.is_available():
        gpu_start = time.time()
        a_gpu = torch.randn(size, size, device='cuda')
        b_gpu = torch.randn(size, size, device='cuda')
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - gpu_start
        
        print(f"Matrix size: {size}x{size}")
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    else:
        print("CUDA není dostupná!")

if __name__ == "__main__":
    benchmark_gpu()
EOF

# Ollama Quick Start
cat > "$EXAMPLES/quick-start/05_ollama_basics.py" << 'EOF'
#!/usr/bin/env python3
"""Ollama Quick Start - Lokální LLM."""
import requests
import json

OLLAMA_URL = "http://localhost:11434"

def chat(model: str, prompt: str) -> str:
    """Chat s Ollama modelem."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def list_models() -> list:
    """Seznam dostupných modelů."""
    response = requests.get(f"{OLLAMA_URL}/api/tags")
    return [m["name"] for m in response.json().get("models", [])]

if __name__ == "__main__":
    print("Dostupné modely:", list_models())
    
    # Chat
    response = chat("llama3.2", "Vysvětli mi co je neural network ve 2 větách.")
    print(f"\nOdpověď: {response}")
EOF

# Fine-tuning Quick Start
cat > "$EXAMPLES/quick-start/06_finetuning_basics.py" << 'EOF'
#!/usr/bin/env python3
"""Fine-tuning Quick Start - LoRA s PEFT."""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Model
model_name = "microsoft/phi-2"  # Malý model pro demo
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

# LoRA konfigurace
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                      # Rank
    lora_alpha=32,            # Alpha
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Které vrstvy adaptovat
)

# Aplikace LoRA
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

print("""
Pro plný fine-tuning použijte:
- datasets pro načtení dat
- Trainer z transformers
- TrainingArguments pro konfiguraci

Příklad datasetu:
dataset = load_dataset("json", data_files="your_data.jsonl")
""")
EOF

# Digital Twin Quick Start
cat > "$EXAMPLES/quick-start/07_digital_twin_basics.py" << 'EOF'
#!/usr/bin/env python3
"""Digital Twin Quick Start - Základní simulace."""
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import List, Dict
import time

@dataclass
class SensorReading:
    """Simulované senzorové čtení."""
    timestamp: float
    temperature: float
    pressure: float
    vibration: float
    status: str

class DigitalTwin:
    """Jednoduchý digitální dvojče stroje."""
    
    def __init__(self, name: str):
        self.name = name
        self.readings: List[SensorReading] = []
        self.state = "normal"
        
    def simulate_sensor(self) -> SensorReading:
        """Simulace senzorových dat."""
        # Normální provoz s náhodným šumem
        temp = 65 + np.random.normal(0, 5)
        pressure = 100 + np.random.normal(0, 10)
        vibration = 0.5 + np.random.exponential(0.1)
        
        # Detekce anomálií
        if temp > 80 or vibration > 1.0:
            status = "warning"
        elif temp > 90 or vibration > 1.5:
            status = "critical"
        else:
            status = "normal"
            
        return SensorReading(
            timestamp=time.time(),
            temperature=round(temp, 2),
            pressure=round(pressure, 2),
            vibration=round(vibration, 3),
            status=status
        )
    
    def update(self) -> Dict:
        """Aktualizace stavu dvojčete."""
        reading = self.simulate_sensor()
        self.readings.append(reading)
        self.state = reading.status
        return asdict(reading)
    
    def predict_maintenance(self) -> str:
        """Jednoduchá predikce údržby."""
        if len(self.readings) < 10:
            return "Nedostatek dat pro predikci"
        
        recent = self.readings[-10:]
        avg_temp = np.mean([r.temperature for r in recent])
        avg_vib = np.mean([r.vibration for r in recent])
        
        if avg_temp > 75 or avg_vib > 0.8:
            return "Doporučena údržba do 7 dní"
        return "Stav OK, žádná údržba není nutná"

if __name__ == "__main__":
    # Vytvoření digitálního dvojčete
    twin = DigitalTwin("Pump-001")
    
    # Simulace 20 čtení
    print(f"Digital Twin: {twin.name}")
    print("-" * 50)
    
    for i in range(20):
        data = twin.update()
        print(f"Reading {i+1}: T={data['temperature']}°C, "
              f"P={data['pressure']}kPa, V={data['vibration']}, "
              f"Status={data['status']}")
        time.sleep(0.1)
    
    print("-" * 50)
    print(f"Predikce: {twin.predict_maintenance()}")
EOF

chown -R "$USER_REAL:$USER_REAL" "$EXAMPLES"

#===============================================================================
# INDEX SOUBOR
#===============================================================================
cat > "$EXAMPLES/README.md" << 'EOF'
# AI Examples & Tutorials

## Struktura

### NVIDIA
- `nvidia/cuda-samples` - Oficiální CUDA příklady
- `nvidia/tensorrt` - TensorRT optimalizace
- `nvidia/triton-tutorials` - Inference server
- `nvidia/nemo` - Conversational AI
- `nvidia/modulus` - Physics ML
- `nvidia/rapids-notebooks` - GPU data science

### Hugging Face
- `huggingface/transformers` - Transformers příklady
- `huggingface/diffusers` - Stable Diffusion, etc.
- `huggingface/peft` - LoRA, QLoRA fine-tuning
- `huggingface/trl` - RLHF training
- `huggingface/notebooks` - Oficiální tutoriály

### IBM
- `ibm/granite-code` - Granite code modely
- `ibm/watsonx-samples` - watsonx.ai příklady
- `ibm/generative-ai` - IBM GenAI SDK

### PyTorch & TensorFlow
- `pytorch/examples` - Oficiální příklady
- `pytorch/tutorials` - Tutoriály
- `tensorflow/examples` - TF příklady
- `tensorflow/keras-examples` - Keras příklady

### LLM & Fine-tuning
- `llm/ollama` - Ollama
- `llm/litgpt` - LitGPT fine-tuning
- `llm/axolotl` - Axolotl fine-tuning
- `llm/vllm` - vLLM inference
- `langchain/langchain` - LangChain

### Computer Vision
- `computer-vision/yolo` - YOLO
- `computer-vision/detectron2` - Detectron2
- `computer-vision/segment-anything` - SAM

### Digital Twin & Simulation
- `digital-twin/omniverse-kit` - NVIDIA Omniverse
- `digital-twin/isaac-gym` - Isaac Sim
- `digital-twin/mujoco` - MuJoCo

### Quick Start (vlastní příklady)
- `quick-start/01_pytorch_basics.py`
- `quick-start/02_huggingface_basics.py`
- `quick-start/03_langchain_basics.py`
- `quick-start/04_cuda_basics.py`
- `quick-start/05_ollama_basics.py`
- `quick-start/06_finetuning_basics.py`
- `quick-start/07_digital_twin_basics.py`

## Spuštění

```bash
# Aktivace AI prostředí
conda activate ai

# Spuštění quick-start příkladu
python ~/Examples/quick-start/01_pytorch_basics.py
```
EOF

ok "========== PŘÍKLADY STAŽENY =========="
log "Příklady jsou v: ~/Examples/"
log "Quick-start: ~/Examples/quick-start/"

