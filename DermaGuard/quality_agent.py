"""Image Quality Agent (rule-based, explainable).

Checks blur, brightness, contrast, glare, dark corners, occlusion and hair.
Thresholds come from percentiles of TRAINING images only (see fit_thresholds).
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN_DIR = ROOT / "health-training-jiit" / "Health" / "training" / "images"
EVAL_DIR = ROOT / "health-eval-jiit" / "Health" / "eval" / "images"
OUT_DIR = HERE / "outputs"
THR_PATH = OUT_DIR / "thresholds.json"

# penalty weight of each check (sum > 1 on purpose; score is clipped to 0-1)
WEIGHTS = {"blur": 0.35, "brightness": 0.2, "contrast": 0.15, "glare": 0.15,
           "dark_corner": 0.15, "occlusion": 0.15, "hair": 0.10}

JUSTIFY = {
    "blur_min": "5th percentile of Laplacian variance on training images: sharper than the blurriest 5%.",
    "blur_poor": "1st percentile of Laplacian variance: below this the image is among the blurriest 1% (unusable).",
    "brightness_low": "5th percentile of mean gray level: darker than 95% of training images.",
    "brightness_high": "95th percentile of mean gray level: brighter than 95% of training images.",
    "contrast_min": "5th percentile of gray std-dev: flatter than 95% of training images.",
    "glare_max": "95th percentile of fraction of bright low-saturation pixels (specular glare).",
    "dark_corner_max": "95th percentile of dark pixels in the 4 corner patches (vignette / dermoscope edge).",
    "occlusion_max": "95th percentile of pixels that are almost pure black or pure white.",
    "hair_max": "95th percentile of thin dark line pixels found by blackhat filter (hair / ruler).",
}
METRIC_FOR = {"blur_min": "blur", "blur_poor": "blur", "brightness_low": "brightness",
              "brightness_high": "brightness", "contrast_min": "contrast", "glare_max": "glare_ratio",
              "dark_corner_max": "dark_corner_ratio", "occlusion_max": "occlusion_ratio",
              "hair_max": "hair_ratio"}


def _load(src):
    if isinstance(src, np.ndarray):
        return src
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"Cannot read image: {src}")
    return img


class ImageQualityAgent:
    def __init__(self, thresholds_path=THR_PATH, auto_fit=True):
        self.thresholds_path = Path(thresholds_path)
        self.thr = None
        if self.thresholds_path.exists():
            self.thr = json.loads(self.thresholds_path.read_text())["thresholds"]
        elif auto_fit:
            self.fit_thresholds(self.train_paths())

    # ---------- helpers ----------
    @staticmethod
    def train_paths():
        return sorted(TRAIN_DIR.glob("*.jpg"))

    @staticmethod
    def compute_metrics(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = gray.shape
        ph, pw = int(0.15 * h), int(0.15 * w)
        corners = [gray[:ph, :pw], gray[:ph, -pw:], gray[-ph:, :pw], gray[-ph:, -pw:]]
        corner_px = np.concatenate([c.ravel() for c in corners])
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
        return {
            "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "glare_ratio": float(((hsv[..., 2] > 240) & (hsv[..., 1] < 40)).mean()),
            "dark_corner_ratio": float((corner_px < 30).mean()),
            "occlusion_ratio": float(((gray < 15) | (gray > 250)).mean()),
            "hair_ratio": float((bh > 10).mean()),
        }

    # ---------- threshold fitting (TRAINING images only) ----------
    def fit_thresholds(self, image_paths, n=1500, seed=42):
        paths = list(image_paths)
        random.Random(seed).shuffle(paths)
        paths = paths[:n]
        rows = [self.compute_metrics(_load(p)) for p in paths]
        df = pd.DataFrame(rows)
        p = lambda col, q: float(np.percentile(df[col], q))
        thr = {
            "blur_min": p("blur", 5), "blur_poor": p("blur", 1),
            "brightness_low": p("brightness", 5), "brightness_high": p("brightness", 95),
            "contrast_min": p("contrast", 5), "glare_max": p("glare_ratio", 95),
            "dark_corner_max": p("dark_corner_ratio", 95), "occlusion_max": p("occlusion_ratio", 95),
            "hair_max": p("hair_ratio", 95),
        }
        dist = {c: {"min": float(df[c].min()), "p1": p(c, 1), "p5": p(c, 5), "median": p(c, 50),
                    "p95": p(c, 95), "p99": p(c, 99), "max": float(df[c].max())} for c in df.columns}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.thresholds_path.write_text(json.dumps({
            "n_images_used": len(paths), "source": "training images only",
            "thresholds": thr, "justification": JUSTIFY, "distribution": dist}, indent=2))
        self.thr = thr
        return thr

    # ---------- assessment ----------
    def assess(self, src):
        m = self.compute_metrics(_load(src))
        t = self.thr
        fails, reasons = {}, []

        if m["blur"] < t["blur_min"]:
            fails["blur"] = True
            reasons.append(f"Low sharpness (blur): Laplacian variance {m['blur']:.1f} < {t['blur_min']:.1f}")
        if m["brightness"] < t["brightness_low"]:
            fails["brightness"] = True
            reasons.append(f"Too dark: mean brightness {m['brightness']:.1f} < {t['brightness_low']:.1f}")
        elif m["brightness"] > t["brightness_high"]:
            fails["brightness"] = True
            reasons.append(f"Too bright: mean brightness {m['brightness']:.1f} > {t['brightness_high']:.1f}")
        if m["contrast"] < t["contrast_min"]:
            fails["contrast"] = True
            reasons.append(f"Low contrast: std {m['contrast']:.1f} < {t['contrast_min']:.1f}")
        if m["glare_ratio"] > t["glare_max"]:
            fails["glare"] = True
            reasons.append(f"Glare: {m['glare_ratio']*100:.2f}% saturated pixels > {t['glare_max']*100:.2f}%")
        if m["dark_corner_ratio"] > t["dark_corner_max"]:
            fails["dark_corner"] = True
            reasons.append(f"Dark corners/vignette: {m['dark_corner_ratio']*100:.1f}% > {t['dark_corner_max']*100:.1f}%")
        if m["occlusion_ratio"] > t["occlusion_max"]:
            fails["occlusion"] = True
            reasons.append(f"Possible obstruction: {m['occlusion_ratio']*100:.1f}% extreme pixels > {t['occlusion_max']*100:.1f}%")
        if m["hair_ratio"] > t["hair_max"]:
            fails["hair"] = True
            reasons.append(f"Heavy hair/ruler artifacts: {m['hair_ratio']*100:.1f}% > {t['hair_max']*100:.1f}%")

        penalty = sum(WEIGHTS[k] for k in fails)
        score = float(np.clip(1 - penalty, 0, 1))
        if m["blur"] < t["blur_poor"] or len(fails) >= 2 or score < 0.5:
            status = "POOR"
        elif not fails and score >= 0.8:
            status = "GOOD"
        else:
            status = "ACCEPTABLE"
        if not reasons:
            reasons = ["All quality checks passed"]
        return {"quality_score": round(score, 3), "status": status, "reasons": reasons,
                "failed_checks": sorted(fails), "metrics": {k: round(v, 4) for k, v in m.items()}}

    def assess_folder(self, folder, out_csv):
        rows = []
        for p in sorted(Path(folder).glob("*.jpg")):
            r = self.assess(p)
            rows.append({"image_id": p.stem, "quality_score": r["quality_score"], "status": r["status"],
                         "reasons": "; ".join(r["reasons"]), **r["metrics"]})
        df = pd.DataFrame(rows)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return df


if __name__ == "__main__":
    agent = ImageQualityAgent()
    info = json.loads(THR_PATH.read_text())
    print("THRESHOLDS (from", info["n_images_used"], "training images):")
    for k, v in info["thresholds"].items():
        print(f"  {k:16s} {v:10.4f}   {info['justification'][k]}")
    paths = agent.train_paths()
    rnd = random.Random(0)
    sample = rnd.sample(paths, 20)
    print("\n3 example outputs:")
    for p in sample[:3]:
        print(p.name, json.dumps(agent.assess(p), indent=2))
    print("\nStatus counts over 300 random training images:")
    res = [agent.assess(p)["status"] for p in rnd.sample(paths, 300)]
    print(pd.Series(res).value_counts().to_string())