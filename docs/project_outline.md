# Project Outline

## Objective

Build a multi-class traffic sign image classifier using the GTSRB dataset and compare baseline deep learning models with pretrained computer vision architectures.

## Workflow

1. Understand the dataset and inspect its class distribution.
2. Load images and labels and verify the label-to-image mapping.
3. Preprocess images and define consistent data splits.
4. Train baseline models (MLP and Custom CNN).
5. Fine-tune pretrained models (MobileNetV2, EfficientNet-B0, ResNet50, DenseNet121, and VGG16).
6. Evaluate all experiments using the same evaluation approach.
7. Compare results and document findings.

## Evaluation

Report accuracy, precision, recall, and F1-score. Include confusion matrices and training/validation curves where relevant.

## Experiment Fairness

Use the same data split and comparable evaluation settings for all models. Clearly document any model-specific input size, augmentation, frozen layers, and fine-tuning strategy.
