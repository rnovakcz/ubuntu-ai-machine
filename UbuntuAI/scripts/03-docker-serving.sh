#!/bin/bash
#===============================================================================
# 03-docker-serving.sh - Docker + Model Serving (Ollama, vLLM, Triton)
# Ubuntu 25.10 AI Development Environment
#===============================================================================

set -e
exec > >(tee -a /var/log/ai-setup.log) 2>&1

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok() { echo -e "${GREEN}[OK]${NC} $1"; }

[[ $EUID -ne 0 ]] && { echo "Spusťte jako root (sudo)"; exit 1; }
USER_REAL=${SUDO_USER:-$USER}
HOME_REAL=$(getent passwd "$USER_REAL" | cut -d: -f6)

log "========== DOCKER + MODEL SERVING =========="

#--- Docker ---
log "Instalace Docker..."
apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
apt install -y ca-certificates curl gnupg

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu noble stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

usermod -aG docker "$USER_REAL"
systemctl enable docker --now

#--- NVIDIA Container Runtime ---
log "Konfigurace NVIDIA Container Runtime..."
nvidia-ctk runtime configure --runtime=docker || true

cat > /etc/docker/daemon.json << 'EOF'
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "log-driver": "json-file",
    "log-opts": {"max-size": "50m", "max-file": "3"}
}
EOF
systemctl restart docker

#--- Ollama ---
log "Instalace Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Ollama service s GPU
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
EOF
systemctl daemon-reload
systemctl enable ollama --now

#--- Docker images ---
log "Stahování AI Docker images..."
docker pull ollama/ollama:latest || true
docker pull nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04 || true
docker pull pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime || true
docker pull ghcr.io/huggingface/text-generation-inference:latest || true
docker pull nvcr.io/nvidia/tritonserver:24.08-py3 || true

#--- Docker Compose šablony ---
log "Vytváření Docker Compose šablon..."
mkdir -p "$HOME_REAL/ai-workspace/docker"

cat > "$HOME_REAL/ai-workspace/docker/ai-stack.yml" << 'EOF'
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    runtime: nvidia
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    volumes:
      - webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    container_name: minio
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    restart: unless-stopped

volumes:
  ollama_data:
  webui_data:
  qdrant_data:
  minio_data:
EOF

cat > "$HOME_REAL/ai-workspace/docker/tgi.yml" << 'EOF'
version: '3.8'
# Text Generation Inference server
services:
  tgi:
    image: ghcr.io/huggingface/text-generation-inference:latest
    container_name: tgi
    runtime: nvidia
    ports:
      - "8080:80"
    volumes:
      - ../models:/models
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - MODEL_ID=microsoft/Phi-3-mini-4k-instruct
      - QUANTIZE=bitsandbytes-nf4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF

cat > "$HOME_REAL/ai-workspace/docker/triton.yml" << 'EOF'
version: '3.8'
# NVIDIA Triton Inference Server
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.08-py3
    container_name: triton
    runtime: nvidia
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ../models/triton:/models
    command: tritonserver --model-repository=/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF

#--- Utility skripty ---
cat > "$HOME_REAL/ai-workspace/scripts/start-ai-stack.sh" << 'EOF'
#!/bin/bash
cd ~/ai-workspace/docker
docker compose -f ai-stack.yml up -d
echo "Services running:"
echo "  Ollama:     http://localhost:11434"
echo "  Open WebUI: http://localhost:3000"
echo "  Qdrant:     http://localhost:6333"
echo "  MinIO:      http://localhost:9001"
EOF
chmod +x "$HOME_REAL/ai-workspace/scripts/start-ai-stack.sh"

cat > "$HOME_REAL/ai-workspace/scripts/ollama-models.sh" << 'EOF'
#!/bin/bash
# Stažení populárních modelů do Ollama
echo "Stahování modelů..."
ollama pull llama3.2
ollama pull mistral
ollama pull codellama
ollama pull phi3
ollama pull nomic-embed-text
echo "Hotovo! Seznam: ollama list"
EOF
chmod +x "$HOME_REAL/ai-workspace/scripts/ollama-models.sh"

chown -R "$USER_REAL:$USER_REAL" "$HOME_REAL/ai-workspace"

#--- lazydocker ---
log "Instalace lazydocker (TUI)..."
curl -fsSL https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash

ok "========== DOCKER + SERVING HOTOVO =========="
log "Ollama běží na: http://localhost:11434"
log "Spuštění stacku: ~/ai-workspace/scripts/start-ai-stack.sh"
log "DŮLEŽITÉ: Odhlaste se pro aktivaci docker skupiny"

