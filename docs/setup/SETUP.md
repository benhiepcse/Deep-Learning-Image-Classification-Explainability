# H01 - Local Environment Setup

## Environment

- OS: Ubuntu 22.04.5 LTS on WSL2
- Python: 3.10.12
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- CUDA runtime: 12.8

## Create the virtual environment

    python3 -m venv .venv
    source .venv/bin/activate

## Install dependencies

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

The CUDA-enabled PyTorch packages were originally installed with:

    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

## Verify CUDA

    python -c "import torch; print(torch.cuda.is_available())"

Expected result on the validated machine: True.

## Evidence

- environment.txt: operating system, hardware and dependency information.
- smoke_test.txt: GPU tensor operation and verification result.

The environment was validated for task S01-01.

Note: Reinstallation from requirements.txt has not yet been
independently tested in a clean environment.
