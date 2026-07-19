# MODO: Moment-Optimized Distribution Learning for Pre-Trained Model-Based Class-Incremental Learning

## 🎉 Introduction

MODO is a training-free, closed-form approach to class-incremental learning with pre-trained vision-language models. It combines:

1. **Distribution-Oriented Learning (DOL)**: A lightweight linear transport layer that optimizes class-conditional distribution geometry (mean separation relative to variance along inter-class directions) without modifying the frozen backbone.
2. **Gaussian Discriminant Analysis (GDA)**: A closed-form shared-covariance classifier built from the transported moments, requiring no additional classifier training.

MODO achieves state-of-the-art performance across 9 benchmarks with minimal computational overhead.

## 📰 What's New
- Initial release of MODO.

## ☄️ How to Use

### 🕹️ Clone

```bash
git clone https://github.com/QiandingHuang666/MODO-release.git
cd MODO-release
```

### 🗂️ Dependencies

This project uses [uv](https://github.com/astral-sh/uv) for environment management. Key dependencies:

| Package | Version |
|---|---|
| python | 3.8 |
| torch | 1.13.0 |
| torchvision | 0.14.0 |
| open-clip-torch | 2.30.0 |
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| scikit-learn | 1.3.2 |
| pandas | 2.0.3 |
| timm | 1.0.27 |
| tqdm | 4.68.3 |
| pyyaml | 6.0.3 |
| pillow | 10.4.0 |
| matplotlib | 3.7.5 |

Create the environment with uv:

```bash
uv venv --python 3.8 .venv
source .venv/bin/activate
uv pip install torch==1.13.0 torchvision==0.14.0 open-clip-torch==2.30.0 \
    numpy==1.24.4 scipy==1.10.1 scikit-learn==1.3.2 pandas==2.0.3 \
    timm==1.0.27 tqdm==4.68.3 pyyaml==6.0.3 pillow==10.4.0 matplotlib==3.7.5
```

### 🔑 Run Experiment

```bash
# SimpleCIL (Prototype baseline)
python main.py --config=./exps/simplecil/cifar224_10_10.json

# MODO (DOL + GDA)
python main.py --config=./exps/clip+dol+gda/cifar224_10_10.json
```

Available configs (18 datasets × 2 splits):
- `exps/simplecil/` — SimpleCIL baseline (text prototype + cosine similarity)
- `exps/clip+dol+gda/` — MODO (DOL transport + Gaussian discriminant analysis)

### ⚙️ Key Hyperparameters

- **model_name**: `simplegdaclip` for MODO
- **init_cls**: Number of classes in the initial task
- **increment**: Number of classes per incremental task
- **backbone_type**: `clip` (CLIP ViT-B/16)
- **seed**: Random seed for class order shuffling (default: 1993)
- **use_dol**: Enable DOL transport layer (default: True)
- **dol_epoch**: DOL optimization epochs (default: 10)
- **dol_alpha**: Separation ratio sharpness (default: 1.0)

### 🔎 Datasets

We follow the data preprocessing of [RevisitingCIL](https://github.com/zhoudw-zdw/RevisitingCIL) and [C3Box](https://github.com/LAMDA-CL/C3Box). Place datasets under `./data/`:

| Dataset | Classes | Path |
|---|---|---|
| CIFAR100 | 100 | `./data/cifar100/` (auto-download) |
| Aircraft | 100 | `./data/aircraft/{train,test}/` |
| Cars | 196 | `./data/cars/{train,test}/` |
| CUB200 | 200 | `./data/cub/{train,test}/` |
| Food101 | 101 | `./data/food101/{train,test}/` |
| UCF101 | 101 | `./data/ucf/{train,test}/` |
| SUN397 | 397 | `./data/sun/{train,test}/` |
| ObjectNet | 200 | `./data/objectnet/{train,test}/` |
| ImageNet-R | 200 | `./data/imagenet-r/{train,test}/` |

Download links can be found in the [C3Box README](https://github.com/LAMDA-CL/C3Box).

## 👨‍🏫 Acknowledgments

This codebase is built upon [C3Box](https://github.com/LAMDA-CL/C3Box) and [PyCIL](https://github.com/G-U-N/PyCIL).

## 🤗 Contact

If you have any questions, please open an issue or contact the author. Enjoy the code!
