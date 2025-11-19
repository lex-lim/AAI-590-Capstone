#!/bin/bash

set -e

echo "Checking for Python 3.11..."

if command -v python3.12 &>/dev/null; then
    echo "Python 3.11 is already installed."
    sudo apt install -y python3.12 python3.12-venv python3.12-dev
else
    echo "Python 3.11 not found. Attempting installation..."

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian)
                echo "Installing Python 3.12 using apt..."
                sudo apt update
                sudo apt install -y software-properties-common
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt update
                sudo apt install -y python3.12 python3.12-venv python3.12-dev
                ;;
            fedora)
                echo "Installing Python 3.11 using dnf..."
                sudo dnf install -y python3.11 python3.11-devel
                ;;
            arch)
                echo "Installing Python 3.11 using pacman..."
                sudo pacman -Sy --noconfirm python
                ;;
            *)
                echo "Unsupported Linux distribution: $ID"
                echo "Please install Python 3.11 manually or use pyenv."
                exit 1
                ;;
        esac
    else
        echo "Unable to detect OS. Please install Python 3.11 manually."
        exit 1
    fi
fi

if [ -d ".venv" ]; then
    echo ".venv already exists. Skipping virtual environment creation."
else
    echo "Creating virtual environment with Python 3.11..."
    python3.12 -m venv .venv
fi

echo "Activating virtual environment and installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip

if [ -f "src/requirements.txt" ]; then
    pip install -r src/requirements.txt
else
    echo "No requirements.txt found in src/. Skipping package installation."
fi

echo "Setup complete. To activate the environment, run:"
echo "    source .venv/bin/activate"
