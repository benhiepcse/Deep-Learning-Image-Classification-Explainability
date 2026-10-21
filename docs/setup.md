# Local setup status

This document records preparation steps. A command appearing here is not proof
that it has run successfully; verified outputs belong in the environment
manifest.

## Current observed state

- Repository: present
- Operating system: Windows
- Shell used for the initial audit: PowerShell 7.6.5
- GPU detected by `nvidia-smi`: NVIDIA GeForce RTX 4060 Laptop GPU
- GPU memory reported by `nvidia-smi`: 8188 MiB
- Current `python`: MSYS2 Python 3.14.6
- Current Python has no `pip`, `torch`, or `torchvision`
- Supported project Python environment: not created yet

## Pending preparation

1. Select and install a project-compatible Python version.
2. Create a repository-local virtual environment.
3. Install and verify a compatible PyTorch and torchvision combination.
4. Record exact versions in `reports/templates/environment_manifest.md`.
5. Run a tensor/device smoke test written by the project author.
6. Run a measured batch pilot before estimating the E01-E04 compute budget.
