#!/bin/bash
#===============================================================================
# install.sh - Hlavní instalační skript
# Ubuntu 25.10 AI Development Environment
#
# Použití:
#   chmod +x install.sh
#   sudo ./install.sh [--all|--basic|--models]
#===============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     🤖 Ubuntu AI Development Environment Setup 🤖            ║"
    echo "║                                                              ║"
    echo "║  CUDA • PyTorch • TensorFlow • Hugging Face • Ollama        ║"
    echo "║  Docker • VS Code • LangChain • RAG • Fine-tuning           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err() { echo -e "${RED}[✗]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#--- Kontroly ---
if [[ $EUID -ne 0 ]]; then
    err "Tento skript musí být spuštěn jako root!"
    echo "Použití: sudo $0"
    exit 1
fi

if ! grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
    warn "Tento skript je určen pro Ubuntu. Pokračovat? (y/n)"
    read -r answer
    [[ "$answer" != "y" ]] && exit 1
fi

banner

#--- Výběr instalace ---
show_menu() {
    echo ""
    echo "Vyberte typ instalace:"
    echo ""
    echo "  1) FULL     - Kompletní instalace (doporučeno)"
    echo "                Systém, NVIDIA, Python, Docker, VS Code,"
    echo "                příklady, learning path, modely"
    echo ""
    echo "  2) BASIC    - Základní instalace (bez modelů)"
    echo "                Systém, NVIDIA, Python, Docker, VS Code"
    echo ""
    echo "  3) MODELS   - Pouze stažení modelů"
    echo "                Ollama, Hugging Face, Whisper, YOLO"
    echo ""
    echo "  4) CUSTOM   - Vlastní výběr skriptů"
    echo ""
    echo "  0) EXIT     - Ukončit"
    echo ""
}

run_script() {
    local script="$1"
    local name="$2"
    
    if [[ -f "$SCRIPT_DIR/$script" ]]; then
        log "Spouštím: $name"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        bash "$SCRIPT_DIR/$script"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ok "$name dokončeno"
        echo ""
    else
        err "Skript nenalezen: $script"
    fi
}

install_full() {
    log "Spouštím FULL instalaci..."
    echo ""
    
    run_script "01-system-nvidia.sh" "Systém + NVIDIA + CUDA"
    run_script "02-python-ai.sh" "Python + AI knihovny"
    run_script "03-docker-serving.sh" "Docker + Model Serving"
    run_script "04-gui-vscode.sh" "VS Code + GUI nástroje"
    run_script "05-examples.sh" "Ukázkové repozitáře"
    run_script "06-learning-path.sh" "AI Learning Path"
    
    warn "Stahování modelů může trvat dlouho. Pokračovat? (y/n)"
    read -r answer
    if [[ "$answer" == "y" ]]; then
        run_script "07-download-models.sh" "Lokální modely"
    fi
}

install_basic() {
    log "Spouštím BASIC instalaci..."
    echo ""
    
    run_script "01-system-nvidia.sh" "Systém + NVIDIA + CUDA"
    run_script "02-python-ai.sh" "Python + AI knihovny"
    run_script "03-docker-serving.sh" "Docker + Model Serving"
    run_script "04-gui-vscode.sh" "VS Code + GUI nástroje"
}

install_models() {
    log "Spouštím stahování modelů..."
    echo ""
    run_script "07-download-models.sh" "Lokální modely"
}

install_custom() {
    echo ""
    echo "Dostupné skripty:"
    echo "  1) 01-system-nvidia.sh  - Systém, NVIDIA, CUDA, jazyky"
    echo "  2) 02-python-ai.sh      - Python, Conda, AI knihovny"
    echo "  3) 03-docker-serving.sh - Docker, Ollama, serving"
    echo "  4) 04-gui-vscode.sh     - VS Code, GUI nástroje"
    echo "  5) 05-examples.sh       - Ukázkové repozitáře"
    echo "  6) 06-learning-path.sh  - AI Learning Path"
    echo "  7) 07-download-models.sh- Stažení modelů"
    echo ""
    echo "Zadejte čísla skriptů oddělená mezerou (např: 1 2 4):"
    read -r choices
    
    for choice in $choices; do
        case $choice in
            1) run_script "01-system-nvidia.sh" "Systém + NVIDIA" ;;
            2) run_script "02-python-ai.sh" "Python + AI" ;;
            3) run_script "03-docker-serving.sh" "Docker + Serving" ;;
            4) run_script "04-gui-vscode.sh" "VS Code + GUI" ;;
            5) run_script "05-examples.sh" "Příklady" ;;
            6) run_script "06-learning-path.sh" "Learning Path" ;;
            7) run_script "07-download-models.sh" "Modely" ;;
            *) warn "Neplatná volba: $choice" ;;
        esac
    done
}

#--- Hlavní logika ---
case "${1:-}" in
    --all)
        install_full
        ;;
    --basic)
        install_basic
        ;;
    --models)
        install_models
        ;;
    *)
        show_menu
        read -p "Volba [1-4, 0]: " choice
        case $choice in
            1) install_full ;;
            2) install_basic ;;
            3) install_models ;;
            4) install_custom ;;
            0) exit 0 ;;
            *) err "Neplatná volba"; exit 1 ;;
        esac
        ;;
esac

#--- Závěr ---
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   INSTALACE DOKONČENA! 🎉                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Další kroky:"
echo ""
echo "  1. RESTART systému (nutné pro NVIDIA ovladače)"
echo "     sudo reboot"
echo ""
echo "  2. Po restartu otestujte GPU:"
echo "     nvidia-smi"
echo ""
echo "  3. Aktivujte AI prostředí:"
echo "     conda activate ai"
echo ""
echo "  4. Spusťte test:"
echo "     python ~/ai-workspace/scripts/test-gpu.py"
echo ""
echo "  5. Začněte se učit:"
echo "     cd ~/AI-Learning/01-python-ai"
echo "     python numpy_basics.py"
echo ""
echo "  6. Spusťte AI stack:"
echo "     ~/ai-workspace/scripts/start-ai-stack.sh"
echo ""
echo "Užijte si AI vývoj! 🚀"

