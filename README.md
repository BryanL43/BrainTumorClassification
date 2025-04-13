# BrainTumorClassification

## Overview
This project trains a brain tumor classification model that identifies Gliomas, Meningiomas, Pituitary tumors, or the absence of tumors from MRI images.

## Installation

### Local Setup
1. **Clone the repository:**
```sh
git clone <your-repo-url>
cd <your-repo-name>
```

2. **Create a virtual environment (optional but recommended):**
```
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```

2. **Install dependencies:**
```sh
pip install -r requirements.txt
```

## Specs

### PyTorch
```sh
version: 2.6.0+cu126
```

[PyTorch CUDA 12.6 Source](https://download.pytorch.org/whl/cu126)

### NVIDIA (R) CUDA compiler driver (Optional)
Requires an Nvidia graphics card with CUDA support.
```sh
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0
```

[Download CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive)

## Dataset

### Brain Tumor Classification (MRI) by Sartaj
[View on Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)


## Troubleshooting
- Ensure `CUDA_PATH_V12_6` is configured in your system `Environment Variables.`


## License
This project is licensed under the MIT License.