import os, glob
from collections import Counter
import pandas as pd

ROOT = r"C:\Users\LENOBO PC\JIIT"
TRAIN_BASE = os.path.join(ROOT, "health-training-jiit", "Health", "training")
TR_IMG = os.path.join(TRAIN_BASE, "images")
EVAL_DIR = os.path.join(ROOT, "health-eval-jiit")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_table(path):
    """Load a CSV or Excel file, detecting the real format from the file bytes."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":                      # xlsx is a zip file
        return pd.read_excel(path, engine="openpyxl")
    if head[:4] == b"%PDF":
        raise ValueError("This file is a real PDF, not a table: " + path)
    return pd.read_csv(path)


print("=" * 60)
print("TRAINING FOLDER CONTENTS")
print("=" * 60)
print(os.listdir(TRAIN_BASE))

# ---------- training metadata ----------
cands = glob.glob(os.path.join(TRAIN_BASE, "metadata*"))
print("\nmetadata candidates:", cands)
if not cands:
    raise SystemExit("No metadata file found in " + TRAIN_BASE)
meta_path = cands[0]
print("using:", meta_path)

df = load_table(meta_path)
print("\nmetadata rows:", len(df))
print("columns:", list(df.columns))
print("\nclass counts (dx):")
print(df["dx"].value_counts())
print("\nclass percentages:")
print((df["dx"].value_counts(normalize=True) * 100).round(1))
print("\nunique lesion_id:", df["lesion_id"].nunique())
print("duplicate image_id rows:", df["image_id"].duplicated().sum())
print("missing values per column:")
print(df.isna().sum())

# ---------- training images vs metadata ----------
files = [f for f in os.listdir(TR_IMG) if os.path.splitext(f)[1].lower() in IMG_EXT]
print("\ntraining image files:", len(files))
print("extensions:", Counter(os.path.splitext(f)[1].lower() for f in files))
ids_meta = set(df["image_id"].astype(str))
ids_files = {os.path.splitext(f)[0] for f in files}
print("in metadata but no image file:", len(ids_meta - ids_files))
print("image file but not in metadata:", len(ids_files - ids_meta))

# ---------- eval folder ----------
print("\n" + "=" * 60)
print("EVAL FOLDER")
print("=" * 60)
ext_count = Counter()
img_total = 0
other_files = []
for dp, dn, fn in os.walk(EVAL_DIR):
    imgs_here = [f for f in fn if os.path.splitext(f)[1].lower() in IMG_EXT]
    img_total += len(imgs_here)
    print(dp, "->", len(fn), "files,", len(imgs_here), "images")
    for f in fn:
        e = os.path.splitext(f)[1].lower()
        ext_count[e] += 1
        if e not in IMG_EXT:
            other_files.append(os.path.join(dp, f))
print("\ntotal eval images:", img_total)
print("eval files by extension:", dict(ext_count))
print("non-image files (possible labels/metadata/readme):")
for p in other_files:
    print("   ", p)

# if the eval folder has a table, check whether it has labels
for p in other_files:
    if os.path.splitext(p)[1].lower() in {".csv", ".xlsx", ".xls", ""}:
        try:
            t = load_table(p)
            print("\neval table:", p, "| rows:", len(t), "| columns:", list(t.columns))
            print("has 'dx' column (labels):", "dx" in t.columns)
        except Exception as e:
            print("could not read", p, "->", e)

# ---------- image size sample ----------
try:
    from PIL import Image
    sizes = Counter()
    for f in files[:50]:
        with Image.open(os.path.join(TR_IMG, f)) as im:
            sizes[im.size] += 1
    print("\nsample image sizes (first 50 train images):", dict(sizes))
except Exception as e:
    print("size check skipped:", e)

# ---------- GPU ----------
try:
    import torch
    print("\nCUDA available:", torch.cuda.is_available())
except Exception as e:
    print("torch check skipped:", e)