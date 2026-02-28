#!/bin/bash
# setup.sh - Complete project setup script for DQN LunarLander

set -e  # Exit on error

echo "=========================================="
echo "DQN LunarLander Project Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python version
echo ""
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python $PYTHON_VERSION detected"
    
    # Check if version is at least 3.8
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
        print_error "Python 3.8+ required. Please upgrade Python."
        exit 1
    fi
else
    print_error "Python3 not found. Please install Python 3.8+."
    exit 1
fi

# Check for CUDA
echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    print_status "CUDA detected (GPU training available)"
    nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1
else
    print_warning "CUDA not detected. Training will use CPU (slower)."
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
elif
    source venv/bin/activate
else
    print error "Could not activate virtual environment"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_status "pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
# pip install -r requirements.txt
# print_status "Dependencies installed"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_status "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Ensure core packages are present
python -c "import torch, gym, moviepy" 2>/dev/null || print_warning "Some core packages missing (torch, gym, moviepy)"

# Install development dependencies
echo ""
echo "Installing development dependencies..."
pip install pytest pytest-cov flake8 black jupyter pre-commit > /dev/null 2>&1
print_status "Development dependencies installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p models plots videos logs assets/videos
touch models/.gitkeep plots/.gitkeep logs/.gitkeep videos/.gitkeep assets/videos/.gitkeep
print_status "Directories created"

# Download pre-trained model (if available)
echo ""
echo "Checking for pre-trained models..."
if [ -f "models/best_model.pth" ]; then
    print_status "Pre-trained model found"
else
    print_warning "No pre-trained model found. Run 'python train.py' to train."
fi

# Run tests
echo ""
echo "Running tests..."
if pytest test_dqn.py -v --tb=short; then
    print_status "All tests passed"
else
    print_warning "Some tests failed. Check test output for details."
fi

# Check Docker
echo ""
echo "Checking Docker installation..."
if command -v docker &> /dev/null; then
    print_status "Docker detected"
    echo "  You can build with: docker-compose build"
else
    print_warning "Docker not found. Docker support will be limited."
fi

# Make check
echo ""
echo "Checking gifsicle installation..."
if command -v gifsicle &> /dev/null; then
    print_status "gifsicle detected (GIF optimization available)"
else
    print_warning "gifsicle not found. GIF optimization will be skipped."
fi

# Git LFS check
echo ""
echo "Checking Git LFS..."
if command -v git &> /dev/null; then
    if git lfs &> /dev/null; then
        print_status "Git LFS detected"
    else
        print_warning "Git LFS not found. Large model files may not sync correctly."
    fi
fi

# Setup summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Project structure:"
echo "  📁 models/     - Model checkpoints"
echo "  📁 plots/      - Training visualizations"
echo "  📁 videos/     - Recorded episodes"
echo "  📁 logs/       - Training logs"
echo "  📁 assets/videos - GIFs from the evaluated videos"
echo ""
echo "Quick start commands:"
echo "  1. Activate environment:  source venv/bin/activate"
echo "  2. Train DQN:             python train.py"
echo "  3. Train Double DQN:      python train.py --double-dqn"
echo "  4. Evaluate agent:        python evaluate.py --model models/path_to_best_model.pth"
echo "  5. Convert to GIFs:       make convert-gifs"
echo "  5. Run tests:             pytest test_dqn.py -v"
echo "  6. View help:             make help"
echo ""
echo "Docker commands:"
echo "  - Build:                  docker-compose build"
echo "  - Train:                  docker-compose up dqn-training"
echo "  - Evaluate:               docker-compose --profile evaluation up dqn-evaluation"
echo ""
echo "For more information, see README.md"
echo "=========================================="