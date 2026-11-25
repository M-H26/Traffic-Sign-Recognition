# Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset for 43-class traffic sign image classification.

## Getting the Dataset

The dataset is intentionally not stored in this GitHub repository because of its size.

Obtain the GTSRB dataset from its official source or from a copy shared separately by the project team. No Google Drive link is required in this repository.

## Suggested Local Structure

```text
data/
├── raw/        # Original downloaded dataset
├── cleaned/    # Cleaned or reorganized data, if needed
└── processed/  # Prepared data or generated metadata
```

Keep the original dataset files in `raw/`. Do not commit dataset images or large generated files to GitHub.

## Labels

GTSRB contains 43 traffic sign classes. Preserve the original class labels when organizing the images.
