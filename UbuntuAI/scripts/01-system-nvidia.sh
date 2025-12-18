#!/bin/bash
#===============================================================================
# 01-system-nvidia.sh - Základní systém, NVIDIA ovladače, CUDA
# Ubuntu 25.10 AI Development Environment
#
# VERZE KOMPATIBILITY (prosinec 2024):
#   CUDA: 12.4 (nejstabilnější pro PyTorch/TensorFlow)
#   Python: 3.11 (nejlepší AI kompatibilita)
#   Node.js: 22 LTS
#   Go: 1.23.x
#   Java: 21 LTS
#   .NET: 8 LTS
#   Rust: stable
#===============================================================================

set -e
exec > >(tee -a /var/log/ai-setup.log) 2>&1

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

[[ $EUID -ne 0 ]] && { echo "Spusťte jako root (sudo)"; exit 1; }
USER_REAL=${SUDO_USER:-$USER}
HOME_REAL=$(getent passwd "$USER_REAL" | cut -d: -f6)

log "========== SYSTÉM + DEV JAZYKY + NVIDIA =========="

#--- Aktualizace systému ---
log "Aktualizace systému..."
apt update && apt upgrade -y

#--- Základní nástroje ---
log "Instalace základních nástrojů..."
apt install -y \
    build-essential cmake ninja-build pkg-config git git-lfs curl wget \
    htop btop nvtop tmux tree unzip jq ripgrep fd-find fzf \
    vim neovim python3 python3-pip python3-venv python-is-python3 \
    libssl-dev libffi-dev libsqlite3-dev libbz2-dev libreadline-dev \
    libncurses-dev libxml2-dev libxslt1-dev libjpeg-dev libpng-dev \
    libopenblas-dev liblapack-dev libhdf5-dev libsndfile1-dev \
    openssh-server snapd flatpak software-properties-common \
    apt-transport-https ca-certificates gnupg lsb-release

sudo -u "$USER_REAL" git lfs install
flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo || true

#===============================================================================
# PROGRAMOVACÍ JAZYKY
#===============================================================================

#--- Node.js 22 LTS + TypeScript ---
log "Instalace Node.js 22 LTS + TypeScript..."
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs
npm install -g \
    yarn@latest \
    pnpm@latest \
    typescript@~5.6 \
    ts-node@latest \
    tsx@latest \
    @types/node@latest \
    nodemon@latest \
    pm2@latest \
    eslint@latest \
    prettier@latest \
    @tensorflow/tfjs-node-gpu@latest

#--- Rust (stable) ---
log "Instalace Rust stable..."
sudo -u "$USER_REAL" bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable'
sudo -u "$USER_REAL" bash -c 'source ~/.cargo/env && rustup component add rust-analyzer clippy rustfmt'
sudo -u "$USER_REAL" bash -c 'source ~/.cargo/env && cargo install cargo-watch cargo-edit sccache'

#--- Go 1.23.x ---
log "Instalace Go 1.23..."
GO_VERSION="1.23.4"
wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -O /tmp/go.tar.gz
rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tar.gz
rm /tmp/go.tar.gz

sudo -u "$USER_REAL" mkdir -p "$HOME_REAL/go"
cat >> "$HOME_REAL/.bashrc" << 'GOEOF'

# Go
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
GOEOF

#--- Java 21 LTS ---
log "Instalace Java 21 LTS..."
apt install -y openjdk-21-jdk openjdk-21-jre maven gradle

# Deep Java Library (DJL) - Java AI framework
sudo -u "$USER_REAL" mkdir -p "$HOME_REAL/.m2"
cat > "$HOME_REAL/.m2/settings.xml" << 'JAVAXML'
<settings>
  <profiles>
    <profile>
      <id>djl</id>
      <repositories>
        <repository>
          <id>djl.ai</id>
          <url>https://oss.sonatype.org/content/repositories/snapshots/</url>
        </repository>
      </repositories>
    </profile>
  </profiles>
  <activeProfiles>
    <activeProfile>djl</activeProfile>
  </activeProfiles>
</settings>
JAVAXML
chown "$USER_REAL:$USER_REAL" "$HOME_REAL/.m2/settings.xml"

cat >> "$HOME_REAL/.bashrc" << 'JAVAEOF'

# Java
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin
JAVAEOF

#--- .NET 8 LTS ---
log "Instalace .NET 8 LTS..."
wget -q https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb -O /tmp/ms-prod.deb
dpkg -i /tmp/ms-prod.deb && rm /tmp/ms-prod.deb
apt update
apt install -y dotnet-sdk-8.0 aspnetcore-runtime-8.0

# .NET AI nástroje
sudo -u "$USER_REAL" bash << 'DOTNETEOF'
export DOTNET_CLI_TELEMETRY_OPTOUT=1
dotnet tool install -g mlnet
dotnet tool install -g Microsoft.dotnet-interactive
dotnet tool install -g dotnet-ef
# Semantic Kernel templates
dotnet new install Microsoft.SemanticKernel.Templates || true
DOTNETEOF

cat >> "$HOME_REAL/.bashrc" << 'DOTNETBASH'

# .NET
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export PATH=$PATH:$HOME/.dotnet/tools
DOTNETBASH

#===============================================================================
# NVIDIA + CUDA
#===============================================================================

log "Detekce NVIDIA GPU..."
if ! lspci | grep -i nvidia > /dev/null; then
    warn "NVIDIA GPU nenalezena! Přeskakuji CUDA instalaci."
    SKIP_NVIDIA=1
fi

if [[ -z "$SKIP_NVIDIA" ]]; then
    log "Odstraňuji staré NVIDIA ovladače..."
    apt purge -y 'nvidia-*' 'libnvidia-*' 'cuda-*' || true
    apt autoremove -y || true

    log "Instalace NVIDIA CUDA repozitáře..."
    apt install -y linux-headers-$(uname -r) dkms
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb
    apt update

    # CUDA 12.4 - nejlepší kompatibilita s PyTorch 2.5 a TensorFlow 2.18
    log "Instalace NVIDIA Driver 550 + CUDA Toolkit 12.4..."
    apt install -y nvidia-driver-550 cuda-toolkit-12-4 libcudnn9-cuda-12

    log "Instalace NCCL a TensorRT..."
    apt install -y libnccl2 libnccl-dev || true
    apt install -y tensorrt || true

    log "Instalace NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt update && apt install -y nvidia-container-toolkit

    cat > /etc/profile.d/cuda.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    chmod +x /etc/profile.d/cuda.sh
    ln -sf /usr/local/cuda-12.4 /usr/local/cuda

    echo -e "blacklist nouveau\noptions nouveau modeset=0" > /etc/modprobe.d/blacklist-nouveau.conf
    update-initramfs -u
fi

#===============================================================================
# SYSTÉMOVÁ OPTIMALIZACE
#===============================================================================

log "Systémová optimalizace..."
cat > /etc/sysctl.d/99-ai.conf << 'EOF'
vm.swappiness = 10
fs.inotify.max_user_watches = 524288
fs.file-max = 2097152
EOF
sysctl --system

#--- AI workspace ---
log "Vytváření AI workspace..."
AI_DIR="$HOME_REAL/ai-workspace"
sudo -u "$USER_REAL" mkdir -p "$AI_DIR"/{models,datasets,projects,notebooks,scripts}

cat >> "$HOME_REAL/.bashrc" << 'EOF'

# AI Environment
export AI_WORKSPACE="$HOME/ai-workspace"
export HF_HOME="$AI_WORKSPACE/models/huggingface"
export TORCH_HOME="$AI_WORKSPACE/models/torch"
[ -d /usr/local/cuda ] && export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH
EOF

chown -R "$USER_REAL:$USER_REAL" "$HOME_REAL"

ok "========== SYSTÉM + JAZYKY + NVIDIA HOTOVO =========="
echo ""
log "Nainstalované verze:"
log "  Node.js: $(node --version)"
log "  Go: $(/usr/local/go/bin/go version)"
log "  Java: $(java --version 2>&1 | head -1)"
log "  .NET: $(dotnet --version)"
log "  Rust: $(sudo -u $USER_REAL bash -c 'source ~/.cargo/env && rustc --version')"
echo ""
warn "RESTART NUTNÝ pro aktivaci NVIDIA ovladačů!"
