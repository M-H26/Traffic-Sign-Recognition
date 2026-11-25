# Traffic Sign Recognition

Traffic Sign Recognition using Deep Learning and Transfer Learning with PyTorch. Comparing CNN architectures and pretrained models on the GTSRB dataset.

## Project Overview

This project explores multi-class image classification for traffic sign recognition using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which contains 43 traffic sign classes.

The goal is to build and compare baseline neural networks with pretrained computer vision models.

## Models

### Baseline Models
- Multi-Layer Perceptron (MLP)
- Custom Convolutional Neural Network (CNN)

### Transfer Learning Models
- MobileNetV2
- EfficientNet-B0
- ResNet50
- DenseNet121
- VGG16

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Repository Structure

```text
Traffic-Sign-Recognition/
├── data/
├── notebooks/
├── results/
├── docs/
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The project uses the GTSRB dataset. The dataset files are hosted externally on Google Drive and are not stored in this repository.

**Dataset link:** [Add Google Drive link here]

See `data/README.md` for dataset organization and setup instructions.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/M-H26/Traffic-Sign-Recognition.git
   cd Traffic-Sign-Recognition
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset using the link above and follow the instructions in `data/README.md`.

4. Open the notebooks in the `notebooks/` directory.

## Experiments and Evaluation

Models will be compared using classification metrics such as accuracy, precision, recall, and F1-score, along with confusion matrices and training/validation performance.

## Team

This is a collaborative academic project. Team members and individual contributions will be documented here.

## Project Status

In progress.
