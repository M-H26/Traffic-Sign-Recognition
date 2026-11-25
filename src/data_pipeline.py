"""
Shared data pipeline for the GTSRB Traffic Sign Recognition project.

All team members should use this module to keep:
- the same final train/validation/test split,
- the same image size and ImageNet normalization,
- the same training augmentation,
- the same optional CLAHE implementation.

Expected dataset index columns:
    Path, ClassId, Split, SignName

Example:
    from data_pipeline import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        index_path="/content/drive/MyDrive/Project Basics of AI 2026/Processed Data/final_dataset_index.csv",
        root_dir="/content/gtsrb_dataset",
        batch_size=32,
        use_clahe=False
    )
"""

import os
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------
# Shared project configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = 224
NUM_CLASSES = 43
RANDOM_SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EXPECTED_SPLIT_COUNTS = {
    "Train": 31367,
    "Validation": 7842,
    "Test": 12630,
}

REQUIRED_INDEX_COLUMNS = {"Path", "ClassId", "Split", "SignName"}


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_worker(worker_id):
    """Seed NumPy and Python random generators inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------

def extract_dataset(zip_path, extract_dir):
    """
    Extract the original dataset ZIP if Train/ and Test/ are not already present.

    Returns the resolved extraction directory.
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)

    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset ZIP was not found: {zip_path}")

    train_folder = extract_dir / "Train"
    test_folder = extract_dir / "Test"

    if not (train_folder.is_dir() and test_folder.is_dir()):
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting dataset to: {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Dataset extraction completed.")
    else:
        print("Dataset is already extracted.")

    if not (train_folder.is_dir() and test_folder.is_dir()):
        raise FileNotFoundError(
            "Expected Train/ and Test/ folders directly inside extract_dir. "
            "Check the ZIP structure and extract_dir."
        )

    return str(extract_dir)


# ---------------------------------------------------------------------
# OpenCV preprocessing: optional CLAHE
# ---------------------------------------------------------------------

def apply_clahe(image):
    """
    Apply CLAHE to the L (luminance) channel in LAB color space.

    Input and output are PIL RGB images.
    """
    image_array = np.array(image.convert("RGB"))

    lab_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced_rgb)


# ---------------------------------------------------------------------
# Shared transformations
# ---------------------------------------------------------------------

def get_transforms(use_clahe=False):
    """
    Return the shared train and evaluation transformations.

    CLAHE is applied by TrafficSignDataset before these transformations.
    The default is disabled because its benefit must be established through
    a controlled validation experiment.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=8,
            translate=(0.05, 0.05),
            scale=(0.90, 1.10)
        ),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    return train_transform, eval_transform


# ---------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------

class TrafficSignDataset(Dataset):
    """Load GTSRB images from the shared index and return image tensor, label."""

    def __init__(self, dataframe, root_dir, transform=None, use_clahe=False):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.root_dir / str(row["Path"])

        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from error

        if self.use_clahe:
            image = apply_clahe(image)

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["ClassId"])
        return image, label


# ---------------------------------------------------------------------
# Index validation and DataLoader creation
# ---------------------------------------------------------------------

def load_and_validate_index(index_path, check_expected_counts=True):
    """Load the final shared CSV and validate its schema and split integrity."""
    index_path = Path(index_path)

    if not index_path.is_file():
        raise FileNotFoundError(f"Dataset index was not found: {index_path}")

    dataframe = pd.read_csv(index_path)

    missing_columns = REQUIRED_INDEX_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required index columns: {sorted(missing_columns)}")

    if dataframe[list(REQUIRED_INDEX_COLUMNS)].isnull().any().any():
        raise ValueError("The dataset index contains missing values.")

    if dataframe["Path"].duplicated().any():
        raise ValueError("Duplicate image paths were found in the dataset index.")

    valid_splits = {"Train", "Validation", "Test"}
    found_splits = set(dataframe["Split"].unique())
    if found_splits != valid_splits:
        raise ValueError(
            f"Expected splits {valid_splits}, but found {found_splits}."
        )

    class_ids = set(dataframe["ClassId"].astype(int).unique())
    expected_classes = set(range(NUM_CLASSES))
    if class_ids != expected_classes:
        raise ValueError(
            f"Expected class IDs 0-42, but found {sorted(class_ids)}."
        )

    if check_expected_counts:
        actual_counts = dataframe["Split"].value_counts().to_dict()
        if actual_counts != EXPECTED_SPLIT_COUNTS:
            raise ValueError(
                "Split counts do not match the agreed project split. "
                f"Expected {EXPECTED_SPLIT_COUNTS}, got {actual_counts}."
            )

    return dataframe


def get_dataloaders(
    index_path,
    root_dir,
    batch_size=32,
    use_clahe=False,
    num_workers=2,
    seed=RANDOM_SEED,
    check_expected_counts=True,
):
    """
    Create the shared Train, Validation, and Test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataframe = load_and_validate_index(
        index_path=index_path,
        check_expected_counts=check_expected_counts
    )

    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(
            f"Extracted dataset directory was not found: {root_dir}"
        )

    train_transform, eval_transform = get_transforms(use_clahe=use_clahe)

    train_data = dataframe[dataframe["Split"] == "Train"].copy()
    val_data = dataframe[dataframe["Split"] == "Validation"].copy()
    test_data = dataframe[dataframe["Split"] == "Test"].copy()

    train_dataset = TrafficSignDataset(
        train_data, root_dir, train_transform, use_clahe
    )
    val_dataset = TrafficSignDataset(
        val_data, root_dir, eval_transform, use_clahe
    )
    test_dataset = TrafficSignDataset(
        test_data, root_dir, eval_transform, use_clahe
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_loader_args
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_args
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_args
    )

    print("Shared DataLoaders created successfully.")
    print(f"Train:      {len(train_dataset):,} images")
    print(f"Validation: {len(val_dataset):,} images")
    print(f"Test:       {len(test_dataset):,} images")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {batch_size}")
    print(f"CLAHE:      {'Enabled' if use_clahe else 'Disabled'}")

    return train_loader, val_loader, test_loader


def get_class_mapping(index_path):
    """Return the shared {ClassId: SignName} mapping from the final index."""
    dataframe = load_and_validate_index(
        index_path=index_path,
        check_expected_counts=False
    )
    mapping = (
        dataframe[["ClassId", "SignName"]]
        .drop_duplicates()
        .sort_values("ClassId")
    )

    if mapping["ClassId"].duplicated().any():
        raise ValueError("A ClassId is associated with more than one SignName.")

    return dict(zip(mapping["ClassId"].astype(int), mapping["SignName"]))
