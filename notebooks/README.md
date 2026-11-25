# Notebooks

Run the notebooks in numerical order where applicable.

| Notebook | Purpose |
|---|---|
| `00_EDA.ipynb` | Dataset exploration, class distribution, image dimensions, and sample visualization |
| `01_Data_Preprocessing.ipynb` | Loading labels, preprocessing, transformations, and data splitting |
| `02_MLP.ipynb` | MLP baseline experiment |
| `03_Custom_CNN.ipynb` | Custom CNN baseline experiment |
| `04_MobileNetV2.ipynb` | Transfer learning with MobileNetV2 |
| `05_EfficientNetB0.ipynb` | Transfer learning with EfficientNet-B0 |
| `06_ResNet50.ipynb` | Transfer learning with ResNet50 |
| `07_DenseNet121.ipynb` | Transfer learning with DenseNet121 |
| `08_VGG16.ipynb` | Transfer learning with VGG16 |

## Notebook Guidelines

- Keep each notebook focused on one task or model.
- Use consistent train/validation/test splits across experiments.
- Set and document random seeds where applicable.
- Record the model configuration and evaluation metrics.
- Avoid committing datasets, model weights, or large generated outputs.
