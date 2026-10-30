"""
Agentic DermaGuard - Configuration File
Defines all global constants, file system paths, class definitions,
and hyperparameters for Track 2 (Healthcare) - Jaypee Agentic AI Summit 2026.
"""

import os
from pathlib import Path
import torch

# ==============================================================================
# DIRECTORY AND FILE PATHS
# ==============================================================================
# Derive root paths dynamically relative to this file
CONFIG_FILE_PATH = Path(__file__).resolve()
DERMAGUARD_DIR = CONFIG_FILE_PATH.parent
WORKSPACE_DIR = DERMAGUARD_DIR.parent

# Training dataset paths (DO NOT MODIFY DATASET FILES)
TRAIN_ROOT = WORKSPACE_DIR / "health-training-jiit" / "Health" / "training"
TRAIN_METADATA_PATH = TRAIN_ROOT / "metadata.csv"
TRAIN_IMAGES_DIR = TRAIN_ROOT / "images"

# Evaluation dataset paths (DO NOT TOUCH DURING MODEL TRAINING/TUNING)
EVAL_ROOT = WORKSPACE_DIR / "health-eval-jiit" / "Health" / "eval"
EVAL_METADATA_PATH = EVAL_ROOT / "metadata.csv"
EVAL_IMAGES_DIR = EVAL_ROOT / "images"

# Output artifacts
CHECKPOINT_DIR = DERMAGUARD_DIR / "checkpoints"
OUTPUT_DIR = DERMAGUARD_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

THRESHOLDS_PATH = DERMAGUARD_DIR / "thresholds.json"
RESULTS_PATH = OUTPUT_DIR / "results.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
COMPARISON_PATH = OUTPUT_DIR / "comparison.csv"
HOLDOUT_SPLIT_PATH = DERMAGUARD_DIR / "holdout_split.csv"

# Ensure output directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# REPRODUCIBILITY & SPLIT SETTINGS
# ==============================================================================
SEED = 42

# Target ~15% - 20% holdout split.
# Using StratifiedGroupKFold with n_splits=5 yields exactly a 20% holdout fold
# with zero lesion_id overlap and stratified class distribution.
N_SPLITS = 5
HOLDOUT_FOLD = 0

# ==============================================================================
# CLASS DEFINITIONS (HAM10000 - 7 Classes)
# ==============================================================================
# Canonical alphabetical ordering
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions (solar lentigines / seborrheic keratoses)",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas)",
}

# Malignant or high-risk classes where recall is clinically critical
HIGH_RISK_CLASSES = ["mel", "bcc", "akiec"]

# ==============================================================================
# IMAGE & PREPROCESSING PARAMETERS
# ==============================================================================
ORIGINAL_IMAGE_SIZE = (600, 450)  # (width, height)
IMAGE_SIZE = 224                  # standard backbone input resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==============================================================================
# MODEL TRAINING & INFERENCE HYPERPARAMETERS
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 is safest on Windows to prevent multiprocessing child spawn issues
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
COSINE_T_MAX = NUM_EPOCHS

# Backbone architectures to compare fairly
ARCHITECTURES = ["efficientnet_b0", "resnet50", "convnext_tiny"]

# Flag for mixed precision training
USE_AMP = torch.cuda.is_available()
