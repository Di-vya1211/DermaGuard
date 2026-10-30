"""Make small copies of the images (256x192) so the Colab upload is quick.
Originals are never modified.   Run:  python DermaGuard/prep_small.py
Creates JIIT/small_data/ and small_data.zip
"""
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = {
    "train": ROOT / "health-training-jiit" / "Health" / "training",
    "eval": ROOT / "health-eval-jiit" / "Health" / "eval",
}
DST = ROOT / "small_data"
SIZE = (256, 192)  # width, height (keeps the 4:3 shape of 600x450)

for split, src in SRC.items():
    out = DST / split / "images"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((src / "images").glob("*.jpg"))
    for i, f in enumerate(files):
        target = out / f.name
        if not target.exists():
            Image.open(f).convert("RGB").resize(SIZE, Image.LANCZOS).save(target, quality=95)
        if i % 1000 == 0:
            print(split, i, "/", len(files))
    shutil.copy(src / "metadata.csv", DST / split / "metadata.csv")

shutil.make_archive(str(ROOT / "small_data"), "zip", DST)
print("Done ->", ROOT / "small_data.zip")