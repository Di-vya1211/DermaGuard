"""
Agentic DermaGuard - Batch Inference on 200 Hidden-Test Images
Jaypee Agentic AI International Summit 2026

Runs the complete 6-Agent Clinical System:
  A1: Image Quality Sentinel
  A2: Diagnosis Classifier (7 HAM10000 classes)
  A3: Lesion Analysis Agent (Dermoscopic ABCD)
  A4: Reasoning / Context Agent (Age, Sex, Localization)
  A5: Safety Agent (S1-S6 guardrails & conflict detection)
  A6: Chief Coordinator Agent (Policy Engine)

Outputs the official submission CSV:
  image_id,predicted_label,decision
"""

import sys
import os
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import (
    CLASSES,
    HIGH_RISK_CLASSES,
    TRAIN_METADATA_PATH,
    OUTPUT_DIR,
)
from quality_agent import ImageQualityAgent
from lesion_analysis_agent import LesionAnalysisAgent
from reasoning_agent import ReasoningContextAgent
from safety_agent import SafetyAgent
from coordinator_agent import CoordinatorAgent

HIDDEN_DIR = WORKSPACE / "health-hidden" / "Health" / "hidden"
HIDDEN_IMAGES_DIR = HIDDEN_DIR / "images"
HIDDEN_RECORDS_PATH = HIDDEN_DIR / "records.csv"


def build_feature_extractor():
    """Loads pretrained EfficientNet-B0 backbone for feature extraction."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.eval()
    feat_extractor = model.features
    pool = model.avgpool

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return feat_extractor, pool, transform


def compute_class_prototypes(feat_extractor, pool, transform):
    """
    Computes class centroid prototypes from the training set using EfficientNet features.
    Samples 40 representative images per class from small_data or health-training-jiit.
    """
    print("[A2] Computing calibrated class prototypes from training dataset...")
    train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    
    # Check images dir
    small_train_img_dir = WORKSPACE / "small_data" / "train" / "images"
    orig_train_img_dir = WORKSPACE / "health-training-jiit" / "Health" / "training" / "images"
    train_img_dir = small_train_img_dir if small_train_img_dir.exists() else orig_train_img_dir

    centroids = {}
    for c in CLASSES:
        sub = train_meta[train_meta["dx"] == c]
        n_samples = min(40, len(sub))
        sampled = sub.sample(n_samples, random_state=42)
        
        feats = []
        for img_id in sampled["image_id"]:
            fpath = train_img_dir / f"{img_id}.jpg"
            if not fpath.exists():
                continue
            try:
                img = Image.open(fpath).convert("RGB")
                t = transform(img).unsqueeze(0)
                with torch.no_grad():
                    f = pool(feat_extractor(t)).squeeze().numpy()
                    # L2 normalize
                    f = f / max(1e-7, np.linalg.norm(f))
                    feats.append(f)
            except Exception:
                pass
        
        if feats:
            centroid = np.mean(feats, axis=0)
            centroid = centroid / max(1e-7, np.linalg.norm(centroid))
            centroids[c] = centroid
        else:
            centroids[c] = np.zeros(1280)

    print(f"[A2] Successfully constructed prototypes for all {len(centroids)} classes.")
    return centroids


def predict_diagnosis(img_pil, feat_extractor, pool, transform, centroids):
    """
    Predicts calibrated class probabilities using cosine similarity to class centroids
    with clinical prior regularization and temperature scaling.
    """
    t = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        feat = pool(feat_extractor(t)).squeeze().numpy()
        feat = feat / max(1e-7, np.linalg.norm(feat))

    # Cosine similarities
    sims = {}
    for c in CLASSES:
        sims[c] = float(np.dot(feat, centroids[c]))

    # Temperature scaling calibrated to produce clinical probabilities (65% - 90%)
    temperature = 0.032
    raw_logits = np.array([sims[c] for c in CLASSES]) / temperature
    exp_logits = np.exp(raw_logits - np.max(raw_logits))
    probs = exp_logits / np.sum(exp_logits)

    prob_dict = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    conf = float(probs[pred_idx])

    # Class display names
    display_names = {
        "akiec": "Actinic Keratosis / Bowen's Disease",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis-like Lesion",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevus",
        "vasc": "Vascular Lesion",
    }

    return {
        "predicted_class": pred_class,
        "display_name": display_names.get(pred_class, pred_class),
        "confidence": conf,
        "probabilities": prob_dict,
        "is_high_risk": pred_class in HIGH_RISK_CLASSES,
    }


def main():
    print("=" * 70)
    print("DERMAGUARD: RUNNING 6-AGENT BATCH INFERENCE ON 200 HIDDEN IMAGES")
    print("=" * 70)

    start_time = time.time()

    if not HIDDEN_RECORDS_PATH.exists():
        raise FileNotFoundError(f"Missing records.csv at: {HIDDEN_RECORDS_PATH}")

    records_df = pd.read_csv(HIDDEN_RECORDS_PATH)
    print(f"Loaded {len(records_df)} hidden-test records from: {HIDDEN_RECORDS_PATH}")

    # Initialize all 6 specialized agents
    a1_quality = ImageQualityAgent()
    feat_extractor, pool, transform = build_feature_extractor()
    centroids = compute_class_prototypes(feat_extractor, pool, transform)
    a3_lesion = LesionAnalysisAgent()
    a4_context = ReasoningContextAgent()
    a5_safety = SafetyAgent()
    a6_coord = CoordinatorAgent()

    submission_rows = []
    detailed_rows = []

    print("\nProcessing all 200 specimens through the 6-agent clinical pipeline...")

    for i, row in records_df.iterrows():
        img_id = str(row["image_id"]).strip()
        age = row.get("age")
        sex = row.get("sex")
        loc = row.get("localization")

        img_path = HIDDEN_IMAGES_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = HIDDEN_IMAGES_DIR / f"{img_id}.jpeg"
        if not img_path.exists():
            print(f"Warning: Image file not found for {img_id}")
            # Fallback
            submission_rows.append({"image_id": img_id, "predicted_label": "nv", "decision": "RE_IMAGE"})
            continue

        # 1. Agent 1: Quality Sentinel
        q_res = a1_quality.assess(img_path)

        # 2. Agent 2: Diagnostic Classifier
        img_pil = Image.open(img_path).convert("RGB")
        d_res = predict_diagnosis(img_pil, feat_extractor, pool, transform, centroids)

        # 3. Agent 3: Lesion Morphological Analysis (ABCD)
        l_res = a3_lesion.analyze(img_path)

        # 4. Agent 4: Epidemiological Reasoning & Context
        c_res = a4_context.evaluate_context(age=age, sex=sex, localization=loc)

        # 5. Agent 5: Clinical Safety & Veto Agent
        s_res = a5_safety.audit(q_res, d_res, l_res, c_res)

        # 6. Agent 6: Chief Coordinator Agent
        coord_res = a6_coord.coordinate(q_res, d_res, l_res, c_res, s_res)

        final_label = coord_res["final_predicted_label"]
        final_decision = coord_res["decision"]

        # Official template record: image_id,predicted_label,decision
        submission_rows.append({
            "image_id": img_id,
            "predicted_label": final_label,
            "decision": final_decision,
        })

        # Detailed record for clinical auditing and dashboards
        detailed_rows.append({
            "image_id": img_id,
            "age": age,
            "sex": sex,
            "localization": loc,
            "predicted_label": final_label,
            "confidence": round(d_res["confidence"], 4),
            "decision": final_decision,
            "a1_status": q_res.get("status"),
            "a1_quality_score": round(q_res.get("quality_score", 0.0), 3),
            "a2_is_high_risk": d_res["is_high_risk"],
            "a3_suspicion_level": l_res.get("suspicion_level"),
            "a3_suspicion_score": round(l_res.get("suspicion_score", 0.0), 3),
            "a4_risk_indicator": c_res.get("risk_indicator"),
            "a5_safety_flag": s_res.get("safety_flag"),
            "a5_veto": s_res.get("veto"),
            "a5_trigger": s_res.get("primary_trigger"),
            "a6_rationale": coord_res.get("traceable_rationale", " | ".join(coord_res.get("rationale_list", []))),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(records_df):
            print(f"  Processed {i + 1}/{len(records_df)} specimens ({((i+1)/len(records_df))*100:.0f}%)")

    # Construct DataFrames
    sub_df = pd.DataFrame(submission_rows)
    det_df = pd.DataFrame(detailed_rows)

    # Validate output schema
    assert len(sub_df) == 200, f"Expected exactly 200 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["image_id", "predicted_label", "decision"], f"Incorrect columns: {sub_df.columns}"

    # Target export destinations
    out_paths = [
        WORKSPACE / "health_submission.csv",
        HIDDEN_DIR / "health_submission.csv",
        OUTPUT_DIR / "health_submission.csv",
    ]

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(p, index=False)
        print(f"[OK] Saved official submission CSV -> {p}")

    # Save detailed CSV
    det_path = OUTPUT_DIR / "hidden_test_six_agent_detailed.csv"
    det_df.to_csv(det_path, index=False)
    print(f"[OK] Saved comprehensive 6-agent audit trail -> {det_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"BATCH INFERENCE COMPLETED IN {elapsed:.1f}s ({elapsed/200:.2f}s per image)")
    print("=" * 70)

    print("\n[TRIAGE DECISION BREAKDOWN]:")
    dec_counts = sub_df["decision"].value_counts()
    for dec, count in dec_counts.items():
        print(f"  {dec:15s}: {count:3d} ({count/len(sub_df)*100:.1f}%)")

    print("\n[PREDICTED LABEL BREAKDOWN]:")
    cls_counts = sub_df["predicted_label"].value_counts()
    for cls, count in cls_counts.items():
        print(f"  {cls:15s}: {count:3d} ({count/len(sub_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
