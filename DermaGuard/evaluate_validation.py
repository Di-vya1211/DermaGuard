"""
Agentic DermaGuard - Validation Dataset Evaluation Runner
Track 2 (Healthcare) - Jaypee Agentic AI Summit 2026

Runs both agents on all 1,942 validation images:
  1. Image Quality Agent: Checks blur, glare, illumination, occlusion, hair/vignette.
  2. Diagnosis Agent: Predicts class and probabilities.
  3. Synthesizes Tri-Decision: AUTO_TRIAGE, REFER, or RE_IMAGE.
  4. Compares predictions against validation ground-truth labels.
  5. Saves comprehensive results to outputs/validation_results.json and outputs/validation_predictions.csv.
"""

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import (
    CLASSES,
    HIGH_RISK_CLASSES,
    EVAL_IMAGES_DIR,
    EVAL_METADATA_PATH,
    OUTPUT_DIR,
)
from quality_agent import ImageQualityAgent
from diagnosis_agent import DiagnosisAgent


def run_full_validation_evaluation():
    print("=" * 70)
    print("RUNNING BOTH AGENTS ON ALL VALIDATION IMAGES")
    print(f"Target directory: {EVAL_IMAGES_DIR}")
    print(f"Metadata path: {EVAL_METADATA_PATH}")
    print("=" * 70)

    quality_agent = ImageQualityAgent()
    diagnosis_agent = DiagnosisAgent()

    meta_df = pd.read_csv(EVAL_METADATA_PATH)
    n_total = len(meta_df)
    print(f"Loaded {n_total} validation records.")

    records = []
    quality_counts = {"GOOD": 0, "ACCEPTABLE": 0, "POOR": 0}
    decision_counts = {"AUTO_TRIAGE": 0, "REFER": 0, "RE_IMAGE": 0}
    quality_failures = {}

    y_true = []
    y_pred = []

    start_time = time.time()
    for idx, row in meta_df.iterrows():
        img_id = row["image_id"]
        true_dx = row["dx"]
        img_path = EVAL_IMAGES_DIR / f"{img_id}.jpg"

        if not img_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        # 1. Quality Agent
        q = quality_agent.assess(bgr)
        quality_counts[q["status"]] = quality_counts.get(q["status"], 0) + 1
        for f in q["failed_checks"]:
            quality_failures[f] = quality_failures.get(f, 0) + 1

        # 2. Diagnosis Agent
        d = diagnosis_agent.predict(bgr, image_id=img_id)

        # 3. Tri-Decision Synthesis
        triage = DiagnosisAgent.compute_tri_decision(q, d)
        decision_counts[triage["decision"]] = decision_counts.get(triage["decision"], 0) + 1

        y_true.append(true_dx)
        y_pred.append(d["predicted_class"])

        records.append({
            "image_id": img_id,
            "ground_truth": true_dx,
            "predicted_class": d["predicted_class"],
            "confidence": d["confidence"],
            "quality_status": q["status"],
            "quality_score": q["quality_score"],
            "final_decision": triage["decision"],
            "failed_checks": ", ".join(q["failed_checks"]) if q["failed_checks"] else "None",
            "is_correct": true_dx == d["predicted_class"],
        })

        if (idx + 1) % 400 == 0 or (idx + 1) == n_total:
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1}/{n_total} validation scans ({elapsed:.1f}s)...")

    df_out = pd.DataFrame(records)

    # Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)

    # Malignancy Sensitivity
    high_risk_true = [t in HIGH_RISK_CLASSES for t in y_true]
    high_risk_pred = [p in HIGH_RISK_CLASSES for p in y_pred]
    mal_sens = float(np.sum(np.array(high_risk_true) & np.array(high_risk_pred)) / np.sum(high_risk_true))

    results_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_validation_images": len(records),
        "classification_performance": {
            "top1_accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(cr["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(cr["weighted avg"]["f1-score"]), 4),
            "malignancy_sensitivity": round(float(mal_sens), 4),
            "per_class": {
                c: {
                    "precision": round(cr[c]["precision"], 4),
                    "recall": round(cr[c]["recall"], 4),
                    "f1": round(cr[c]["f1-score"], 4),
                    "support": cr[c]["support"]
                }
                for c in CLASSES if c in cr
            }
        },
        "image_quality_outcomes": {
            "good_count": quality_counts.get("GOOD", 0),
            "acceptable_count": quality_counts.get("ACCEPTABLE", 0),
            "poor_count": quality_counts.get("POOR", 0),
            "good_pct": round(quality_counts.get("GOOD", 0) / len(records) * 100, 1),
            "acceptable_pct": round(quality_counts.get("ACCEPTABLE", 0) / len(records) * 100, 1),
            "poor_pct": round(quality_counts.get("POOR", 0) / len(records) * 100, 1),
            "pass_rate_pct": round((quality_counts.get("GOOD", 0) + quality_counts.get("ACCEPTABLE", 0)) / len(records) * 100, 1),
            "defect_frequencies": quality_failures,
        },
        "tri_decision_distribution": {
            "AUTO_TRIAGE": decision_counts.get("AUTO_TRIAGE", 0),
            "REFER": decision_counts.get("REFER", 0),
            "RE_IMAGE": decision_counts.get("RE_IMAGE", 0),
            "AUTO_TRIAGE_pct": round(decision_counts.get("AUTO_TRIAGE", 0) / len(records) * 100, 1),
            "REFER_pct": round(decision_counts.get("REFER", 0) / len(records) * 100, 1),
            "RE_IMAGE_pct": round(decision_counts.get("RE_IMAGE", 0) / len(records) * 100, 1),
        }
    }

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_csv_path = OUTPUT_DIR / "validation_predictions.csv"
    val_json_path = OUTPUT_DIR / "validation_results.json"

    df_out.to_csv(val_csv_path, index=False)
    with open(val_json_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("VALIDATION BENCHMARK COMPLETED")
    print("=" * 70)
    print(f"Top-1 Accuracy: {acc*100:.2f}% | Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"Malignancy Sensitivity: {mal_sens*100:.2f}%")
    print(f"Quality Pass Rate: {results_summary['image_quality_outcomes']['pass_rate_pct']}%")
    print(f"Tri-Decisions: {results_summary['tri_decision_distribution']}")
    print(f"Saved predictions CSV to: {val_csv_path}")
    print(f"Saved results summary to: {val_json_path}")

    return results_summary


if __name__ == "__main__":
    run_full_validation_evaluation()
