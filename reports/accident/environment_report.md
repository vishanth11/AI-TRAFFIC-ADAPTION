# Phase 5 — Accident Detector Environment Report

Generated: 2026-09-02

## Python Environment

- **Virtual environment**: `.venv-accident`
- **Python version**: 3.12.10 (64-bit)
- **OS**: Windows 11 AMD64
- **CPU**: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel
- **CPU threads**: 8
- **CUDA available**: False

## Installed Packages

| Package | Version |
|---------|---------|
| torch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| pyarrow | 25.0.1 |
| pillow | 12.3.0 |
| matplotlib | 3.11.1 |
| pycocotools | 2.0.11 |
| tqdm | 4.70.0 |

## Selection Rationale

Python 3.14.7 was the default interpreter, but PyTorch does not publish Windows wheels for Python 3.14. Python 3.12.10 was available via the Windows `py` launcher, and official CPU-only PyTorch wheels exist for it. Therefore the project uses an isolated `.venv-accident` virtual environment with Python 3.12.

## Verification

- `import torch` succeeds.
- `import torchvision` succeeds.
- `torch.cuda.is_available()` returns `False`.
- `import pycocotools` succeeds.

## Requirements File

Exact pinned dependencies are saved to `requirements-accident.txt`.
