## MNIST Digit Classification with CNN

### 📋 Project Overview

This project implements a complete deep learning pipeline for handwritten digit classification using the MNIST dataset. The implementation features a custom LeNet-like convolutional neural network architecture with comprehensive training, evaluation, and visualization capabilities.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)


### Key Features

- **Custom CNN Architecture**: LeNet-inspired neural network adapted for MNIST

- **Complete Training Pipeline**: Loss tracking, validation, and metrics visualization

- **Advanced Visualization**: Feature maps, filters, and activation monitoring

- **Modular Design**: Reusable Architecture class for easy experimentation

- **Comprehensive Evaluation**: Accuracy metrics and confusion matrix analysis

### 🏗️ Architecture
#### Model Structure
The CNN follows a LeNet-like architecture with modern improvements:

```txt
Input (1×28×28)
    ↓
Conv2d(1→32, kernel=3, padding=1) + ReLU
    ↓
MaxPool2d(kernel=2)  # 28×28 → 14×14
    ↓
Conv2d(32→64, kernel=3, padding=1) + ReLU
    ↓
MaxPool2d(kernel=2)  # 14×14 → 7×7
    ↓
Flatten()  # 64×7×7 = 3136 features
    ↓
Linear(3136→128) + ReLU + Dropout(0.5)
    ↓
Linear(128→10)  # 10 output classes (digits 0-9)
```

### Key Components
- **Input**: 28×28 grayscale images

- **Feature Extraction**: Two convolutional layers with ReLU activation and max pooling

- **Classification**: Two fully connected layers with dropout regularization

- **Output**: 10-class probabilities for digits 0-9

###  Dataset

 - **Source:** `torchvision.datasets.MNIST`

- **Classes**: 10 (digits 0-9)

- **Training Samples**: 60,000

- **Validation Samples**: 10,000

- **Image Size**: 28×28 pixels, grayscale

### Data Preprocessing

```python
transform = Compose([
    ToImage(),                    # Convert to tensor
    ToDtype(torch.float32, scale=True),  # Normalize to [0,1]
    Normalize(mean=(0.5,), std=(0.5,))   # Standardize to [-1,1]
])
```

### Implementation

#### Core Classes

`Architecture` Class

Key Methods:

`train()`: Complete training pipeline

`attach_hooks()`: Register hooks for activation capture

`visualize_filters()`: Display convolutional filters

`visualize_outputs()`: Show feature maps and activations

`plot_losses()`: Training/validation loss curves

#### Training Configurations

#### Hyperparameters
```
LR = 0.01
N_EPOCHS = 15
BATCH_SIZE = 64
OPTIMIZER = SGD with momentum (0.9)
LOSS_FN = CrossEntropyLoss
```

### Quick Start

#### Prerequisites

Python 3.8+
pip package manager

#### Installation

1. Clone the repository then Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements

```bash
pip install -r requirements.txt
```

***

> Note: This implementation demonstrates fundamental deep learning concepts and serves as a solid foundation for more complex computer vision tasks. The modular design allows easy adaptation to similar classification problems.