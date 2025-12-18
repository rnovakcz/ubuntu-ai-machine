#!/bin/bash
#===============================================================================
# 04-gui-vscode.sh - VS Code, GUI nástroje a rozšíření
# Ubuntu 25.10 AI Development Environment
#===============================================================================

set -e
exec > >(tee -a /var/log/ai-setup.log) 2>&1

BLUE='\033[0;34m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok() { echo -e "${GREEN}[OK]${NC} $1"; }

[[ $EUID -ne 0 ]] && { echo "Spusťte jako root (sudo)"; exit 1; }
USER_REAL=${SUDO_USER:-$USER}
HOME_REAL=$(getent passwd "$USER_REAL" | cut -d: -f6)

log "========== VS CODE + GUI NÁSTROJE =========="

#--- VS Code ---
log "Instalace VS Code..."
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/keyrings/microsoft.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list
apt update && apt install -y code

#--- VS Code rozšíření pro AI vývoj ---
log "Instalace VS Code rozšíření..."
sudo -u "$USER_REAL" bash << 'VSCODE'
# Python
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff

# Jupyter
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.jupyter-keymap
code --install-extension ms-toolsai.jupyter-renderers
code --install-extension ms-toolsai.vscode-jupyter-cell-tags

# AI & ML
code --install-extension ms-toolsai.vscode-ai
code --install-extension GitHub.copilot
code --install-extension Continue.continue
code --install-extension TabNine.tabnine-vscode

# Docker & Containers
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-vscode-remote.remote-containers
code --install-extension ms-vscode-remote.remote-ssh

# Git
code --install-extension eamodio.gitlens
code --install-extension mhutchie.git-graph
code --install-extension GitHub.vscode-pull-request-github

# JavaScript/TypeScript
code --install-extension dbaeumer.vscode-eslint
code --install-extension esbenp.prettier-vscode
code --install-extension ms-vscode.vscode-typescript-next

# Go
code --install-extension golang.go

# Rust
code --install-extension rust-lang.rust-analyzer

# C#/.NET
code --install-extension ms-dotnettools.csharp
code --install-extension ms-dotnettools.csdevkit

# Java
code --install-extension vscjava.vscode-java-pack
code --install-extension redhat.java

# CUDA/C++
code --install-extension ms-vscode.cpptools
code --install-extension nvidia.nsight-vscode-edition

# Data & Visualization
code --install-extension RandomFractalsInc.vscode-data-preview
code --install-extension GrapeCity.gc-excelviewer
code --install-extension mechatroner.rainbow-csv

# YAML/JSON/TOML
code --install-extension redhat.vscode-yaml
code --install-extension tamasfe.even-better-toml

# Markdown
code --install-extension yzhang.markdown-all-in-one
code --install-extension bierner.markdown-mermaid

# Themes & Productivity
code --install-extension PKief.material-icon-theme
code --install-extension dracula-theme.theme-dracula
code --install-extension streetsidesoftware.code-spell-checker
code --install-extension usernamehw.errorlens
code --install-extension formulahendry.code-runner

# REST/API
code --install-extension humao.rest-client
code --install-extension rangav.vscode-thunder-client
VSCODE

#--- VS Code settings pro AI ---
log "Konfigurace VS Code..."
sudo -u "$USER_REAL" mkdir -p "$HOME_REAL/.config/Code/User"
cat > "$HOME_REAL/.config/Code/User/settings.json" << 'EOF'
{
    "editor.fontSize": 14,
    "editor.fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
    "editor.fontLigatures": true,
    "editor.formatOnSave": true,
    "editor.minimap.enabled": false,
    "editor.rulers": [80, 120],
    "editor.bracketPairColorization.enabled": true,
    
    "python.defaultInterpreterPath": "~/mambaforge/envs/ai/bin/python",
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    },
    
    "jupyter.askForKernelRestart": false,
    "notebook.cellToolbarLocation": "right",
    "notebook.output.scrolling": true,
    
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.fontSize": 13,
    
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000,
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/node_modules": true,
        "**/*.pyc": true
    },
    
    "workbench.colorTheme": "Dracula",
    "workbench.iconTheme": "material-icon-theme",
    
    "git.autofetch": true,
    "git.confirmSync": false,
    
    "ruff.lint.args": ["--line-length=120"],
    
    "docker.environment": {
        "DOCKER_HOST": "unix:///var/run/docker.sock"
    }
}
EOF
chown -R "$USER_REAL:$USER_REAL" "$HOME_REAL/.config/Code"

#--- JetBrains Mono font ---
log "Instalace JetBrains Mono fontu..."
wget -q https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip -O /tmp/jbmono.zip
unzip -q /tmp/jbmono.zip -d /tmp/jbmono
mkdir -p /usr/share/fonts/truetype/jetbrains
cp /tmp/jbmono/fonts/ttf/*.ttf /usr/share/fonts/truetype/jetbrains/
fc-cache -f
rm -rf /tmp/jbmono /tmp/jbmono.zip

#--- Další GUI nástroje ---
log "Instalace dalších GUI nástrojů..."

# NVIDIA GUI nástroje (pokud je GPU)
if command -v nvidia-smi &> /dev/null; then
    apt install -y nvidia-settings || true
fi

# Postman (API testing)
snap install postman || true

# DBeaver (database GUI)
snap install dbeaver-ce || true

# Obsidian (notes)
snap install obsidian --classic || true

# Flameshot (screenshots)
apt install -y flameshot

# Meld (diff tool)
apt install -y meld

ok "========== VS CODE + GUI HOTOVO =========="
log "VS Code: code ."
log "Rozšíření nainstalována pro: Python, Jupyter, Docker, Git, AI, Go, Rust, C#, Java, CUDA"

