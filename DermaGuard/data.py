"""
Agentic DermaGuard - Data Pipeline
Implements:
1. Metadata loading and validation
2. Stratified + Grouped train/holdout split with zero lesion_id overlap
3. Class imbalance compensation (Balanced Class Weights & WeightedRandomSampler)
4. Domain-specific data augmentation and ImageNet normalization
5. PyTorch Dataset and DataLoader factories
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T

# Import configurations
try:
    from . import config
except ImportError:
    import config


# ==============================================================================
# 1. METADATA LOADING & VALIDATION
# ==============================================================================
def load_metadata(metadata_path: Path, images_dir: Path, is_eval: bool = False) -> pd.DataFrame:
    """
    Loads metadata from CSV, validates columns and image existence,
    and adds an absolute path to each image file.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    df = pd.read_csv(metadata_path)
    
    # Verify core columns
    required_cols = ["image_id", "lesion_id"]
    if not is_eval or "dx" in df.columns:
        required_cols.append("dx")

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {metadata_path}")

    # Build and verify image paths
    df["image_path"] = df["image_id"].apply(
        lambda img_id: str(images_dir / f"{img_id}.jpg")
    )
    
    # Verify all files exist
    missing_files = [p for p in df["image_path"] if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(
            f"Found {len(missing_files)} images referenced in metadata but missing on disk! "
            f"Example: {missing_files[0]}"
        )

    # Encode diagnostic class if available
    if "dx" in df.columns:
        # Validate that all classes belong to HAM10000 7 classes
        invalid_classes = set(df["dx"]) - set(config.CLASSES)
        if invalid_classes:
            raise ValueError(f"Unrecognized class labels in metadata: {invalid_classes}")
        df["label"] = df["dx"].map(config.CLASS_TO_IDX)

    return df


# ==============================================================================
# 2. GROUPED + STRATIFIED HOLDOUT SPLIT
# ==============================================================================
def create_stratified_group_split(
    df: pd.DataFrame,
    n_splits: int = config.N_SPLITS,
    holdout_fold: int = config.HOLDOUT_FOLD,
    seed: int = config.SEED,
    save_path: Optional[Path] = config.HOLDOUT_SPLIT_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a split that is simultaneously:
    1. Stratified by diagnostic class ('dx') to maintain class proportions.
    2. Grouped by patient/lesion identifier ('lesion_id') to prevent data leakage.
    
    Guarantees that no lesion appears in both train and holdout partitions.
    Target holdout size: ~20% (with n_splits=5).
    """
    # Defensive copy
    df = df.copy()

    # StratifiedGroupKFold requires features X, target y, and groups
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    folds = list(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
    train_idx, holdout_idx = folds[holdout_fold]

    df["split"] = "train"
    df.loc[holdout_idx, "split"] = "holdout"

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    holdout_df = df[df["split"] == "holdout"].copy().reset_index(drop=True)

    # Strict assertion for zero lesion overlap
    train_lesions = set(train_df["lesion_id"])
    holdout_lesions = set(holdout_df["lesion_id"])
    overlap = train_lesions.intersection(holdout_lesions)
    if len(overlap) > 0:
        raise AssertionError(
            f"CRITICAL ERROR: Data leakage detected! {len(overlap)} lesion_ids appear in both train and holdout!"
        )

    if save_path:
        # Save split with minimal columns for quick reproducibility
        df[["image_id", "lesion_id", "dx", "split"]].to_csv(save_path, index=False)

    return train_df, holdout_df


# ==============================================================================
# 3. CLASS IMBALANCE COMPENSATION
# ==============================================================================
def compute_class_weights(df: pd.DataFrame, num_classes: int = config.NUM_CLASSES) -> torch.FloatTensor:
    """
    Computes balanced class weights using inverse frequency:
    w_c = total_samples / (num_classes * count_c)
    """
    class_counts = df["label"].value_counts().to_dict()
    total_samples = len(df)
    
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        count = class_counts.get(c, 0)
        if count > 0:
            weights[c] = total_samples / (num_classes * count)
        else:
            weights[c] = 1.0

    return torch.from_numpy(weights).float()


def get_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Constructs a WeightedRandomSampler that samples each class with equal probability
    during training, directly tackling the extreme 67% 'nv' vs 1.1% 'df' imbalance.
    """
    class_counts = df["label"].value_counts().to_dict()
    # Weight per class is 1.0 / count
    class_weights = {cls_idx: 1.0 / count for cls_idx, count in class_counts.items()}
    
    # Assign weight to each individual sample
    sample_weights = df["label"].map(class_weights).values
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ==============================================================================
# 4. TRANSFORMS & DATA AUGMENTATIONS
# ==============================================================================
def get_train_transforms(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    """
    Robust dermoscopic image augmentations:
    - RandomResizedCrop: simulates varying dermatoscope distance and zoom
    - RandomHorizontalFlip & RandomVerticalFlip: skin lesions lack inherent rotational orientation
    - RandomRotation: dermatoscope rotation invariance
    - ColorJitter: handles skin tone and illumination variability
    - ImageNet normalization for transfer learning
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


def get_eval_transforms(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    """
    Deterministic evaluation transforms:
    Resize slightly larger then center-crop to target size.
    """
    resize_dim = int(image_size * 1.14)  # ~256 for 224
    return T.Compose([
        T.Resize((resize_dim, resize_dim)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


# ==============================================================================
# 5. PYTORCH DATASET
# ==============================================================================
class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for Dermoscopic Lesion Images.
    Strictly image-only as required (clinical tabular fields not passed to CNN).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[T.Compose] = None,
        is_labeled: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_labeled = is_labeled and ("label" in self.df.columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, str]]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]

        # Load RGB image via PIL
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"]) if self.is_labeled else -1
        
        meta = {
            "image_id": str(row["image_id"]),
            "lesion_id": str(row.get("lesion_id", "")),
            "dx": str(row.get("dx", ""))
        }

        return image, label, meta


# ==============================================================================
# 6. DATALOADER FACTORY
# ==============================================================================
def get_dataloaders(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_weighted_sampler: bool = True,
    image_size: int = config.IMAGE_SIZE
) -> Tuple[DataLoader, DataLoader, torch.FloatTensor]:
    """
    Constructs train and holdout DataLoaders along with balanced class weights.
    """
    train_transform = get_train_transforms(image_size=image_size)
    holdout_transform = get_eval_transforms(image_size=image_size)

    train_dataset = SkinLesionDataset(train_df, transform=train_transform, is_labeled=True)
    holdout_dataset = SkinLesionDataset(holdout_df, transform=holdout_transform, is_labeled=True)

    class_weights = compute_class_weights(train_df)

    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, holdout_loader, class_weights


# ==============================================================================
# SELF-TEST & VERIFICATION ROUTINE
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DERMAGUARD DATA PIPELINE - VERIFICATION & SMOKE TEST")
    print("=" * 70)

    # 1. Load metadata
    print("\n1. Loading and validating training metadata...")
    df_all = load_metadata(config.TRAIN_METADATA_PATH, config.TRAIN_IMAGES_DIR)
    print(f"   Loaded {len(df_all)} valid image records.")

    # 2. Perform Grouped + Stratified Split
    print("\n2. Performing StratifiedGroupKFold split (Seed 42, 5 splits)...")
    train_df, holdout_df = create_stratified_group_split(df_all)

    print(f"   Train set count:   {len(train_df)} ({len(train_df)/len(df_all)*100:.2f}%)")
    print(f"   Holdout set count: {len(holdout_df)} ({len(holdout_df)/len(df_all)*100:.2f}%)")

    # Verify zero lesion overlap
    train_lesions = set(train_df["lesion_id"])
    holdout_lesions = set(holdout_df["lesion_id"])
    overlap = train_lesions.intersection(holdout_lesions)
    print(f"   Unique lesion IDs in Train:   {len(train_lesions)}")
    print(f"   Unique lesion IDs in Holdout: {len(holdout_lesions)}")
    print(f"   Lesion ID Overlap Count:      {len(overlap)} (Zero confirmed: {len(overlap) == 0})")

    # Print class distributions side-by-side
    print("\n3. Class Distributions Comparison:")
    summary_table = []
    train_counts = train_df["dx"].value_counts()
    holdout_counts = holdout_df["dx"].value_counts()
    total_counts = df_all["dx"].value_counts()

    for cls_name in config.CLASSES:
        tot = total_counts.get(cls_name, 0)
        trn = train_counts.get(cls_name, 0)
        hld = holdout_counts.get(cls_name, 0)
        trn_pct = trn / len(train_df) * 100
        hld_pct = hld / len(holdout_df) * 100
        summary_table.append({
            "Class": cls_name,
            "Total": tot,
            "Train N": trn,
            "Train %": f"{trn_pct:.2f}%",
            "Holdout N": hld,
            "Holdout %": f"{hld_pct:.2f}%",
        })

    print(pd.DataFrame(summary_table).to_string(index=False))

    # 4. Balanced Class Weights
    print("\n4. Balanced Class Weights (Inverse Frequency):")
    class_weights = compute_class_weights(train_df)
    for cls_name, idx in config.CLASS_TO_IDX.items():
        print(f"   {cls_name:>5} (idx {idx}): weight = {class_weights[idx]:.3f}")

    # 5. Smoke Test on ~20 images with DataLoader
    print("\n5. Running Smoke Test on a mini-batch of 20 images...")
    sample_df = train_df.head(20).copy()
    sample_dataset = SkinLesionDataset(
        sample_df,
        transform=get_train_transforms(image_size=config.IMAGE_SIZE),
        is_labeled=True
    )
    sample_loader = DataLoader(sample_dataset, batch_size=20, shuffle=False)

    images, labels, metas = next(iter(sample_loader))
    print(f"   Batch image tensor shape: {images.shape} (Expected: [20, 3, 224, 224])")
    print(f"   Batch labels tensor shape: {labels.shape} (Expected: [20])")
    print(f"   Label values in batch:     {labels.tolist()[:10]}...")
    print(f"   Sample image_ids:          {metas['image_id'][:5]}")
    print(f"   Tensor value range:        [{images.min():.2f}, {images.max():.2f}] (Normalized)")

    assert images.shape == (20, 3, 224, 224), "Mismatch in image tensor shape!"
    assert len(labels) == 20, "Mismatch in labels count!"
    print("\n--> DATA PIPELINE SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)
