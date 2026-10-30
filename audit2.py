import os, json
import pandas as pd

ROOT = r"C:\Users\LENOBO PC\JIIT"
TR = os.path.join(ROOT, "health-training-jiit")
EV = os.path.join(ROOT, "health-eval-jiit")

for folder in (TR, EV):
    for name in ("README.md", "manifest.json"):
        p = os.path.join(folder, name)
        print("=" * 70)
        print(p)
        print("=" * 70)
        with open(p, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        print(txt[:3500])
        if len(txt) > 3500:
            print("... [truncated, total", len(txt), "chars]")

tr = pd.read_csv(os.path.join(TR, "Health", "training", "metadata.csv"))
ev = pd.read_csv(os.path.join(EV, "Health", "eval", "metadata.csv"))

print("\n" + "=" * 70)
print("EVAL CLASS COUNTS")
print(ev["dx"].value_counts())
print("\nEVAL unique lesion_id:", ev["lesion_id"].nunique())

print("\nimage_id overlap train/eval:", len(set(tr.image_id) & set(ev.image_id)))
lo = set(tr.lesion_id) & set(ev.lesion_id)
print("lesion_id overlap train/eval:", len(lo))
print("eval images whose lesion also appears in train:", ev.lesion_id.isin(tr.lesion_id).sum())