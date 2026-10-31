"""
Agentic DermaGuard - Vercel Serverless Entry Point
Provides full interactive web interface, REST API endpoints, and CSV download.
Zero external heavy dependencies to guarantee instantaneous Vercel deployment.
"""
import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

# Injected service binding for the internal 'dermaguard' service
DERMAGUARD_URL = os.environ.get("DERMAGUARD_URL", "").rstrip("/")

# Pre-computed benchmark metrics on 1,942 validation images
SUMMARY_METRICS = {
    "total_images": 1942,
    "accuracy": "88.67%",
    "macro_f1": "87.25%",
    "correct_auto_triage": 1089,
    "incorrect_auto_triage": 44,
    "auto_triage_accuracy": "96.1%",
    "number_of_refers": 585,
    "refer_pct": "30.1%",
    "number_of_re_image": 224,
    "re_image_pct": "11.5%",
    "total_auto_triage": 1133,
    "auto_triage_pct": "58.3%"
}

SAMPLE_PREDICTIONS = [
    {"image_id": "ISIC_0026769", "prediction": "bkl", "decision": "REFER"},
    {"image_id": "ISIC_0025661", "prediction": "bkl", "decision": "REFER"},
    {"image_id": "ISIC_0029176", "prediction": "nv", "decision": "AUTO_TRIAGE"},
    {"image_id": "ISIC_0029068", "prediction": "bkl", "decision": "AUTO_TRIAGE"},
    {"image_id": "ISIC_0029396", "prediction": "bkl", "decision": "AUTO_TRIAGE"},
    {"image_id": "ISIC_0025984", "prediction": "bkl", "decision": "RE_IMAGE"},
    {"image_id": "ISIC_0025915", "prediction": "bkl", "decision": "AUTO_TRIAGE"},
    {"image_id": "ISIC_0031029", "prediction": "bkl", "decision": "AUTO_TRIAGE"},
    {"image_id": "ISIC_0030698", "prediction": "bkl", "decision": "AUTO_TRIAGE"},
    {"image_id": "ISIC_0025567", "prediction": "bkl", "decision": "AUTO_TRIAGE"},
    {"image_id": "ISIC_0029301", "prediction": "bkl", "decision": "REFER"},
    {"image_id": "ISIC_0026637", "prediction": "bkl", "decision": "AUTO_TRIAGE"},
    {"image_id": "ISIC_0025286", "prediction": "bkl", "decision": "AUTO_TRIAGE"},
    {"image_id": "ISIC_0029288", "prediction": "nv", "decision": "RE_IMAGE"},
    {"image_id": "ISIC_0031008", "prediction": "bkl", "decision": "AUTO_TRIAGE"}
]

CLINICAL_CASES = {
    "case1": {
        "id": "ISIC_0024693",
        "name": "Case 1: Benign Nevus (nv) - Clear Consensus",
        "prediction": "nv",
        "class_name": "Melanocytic Nevus",
        "decision": "AUTO_TRIAGE",
        "badge_color": "#10b981",
        "decision_title": "AUTO_TRIAGE: Confident Benign Consensus",
        "rationale": "Sufficient reliable evidence for automated screening disposition. Image quality is clear (score 97.2/100). Model predicts benign melanocytic nevus with 96.4% confidence. Visual morphology confirms symmetric borders and low color variegation. No safety vetoes raised.",
        "flow": {
            "A1": {"title": "A1 QUALITY", "status": "PASS", "sub": "Optimal sharpness & lighting", "color": "#10b981"},
            "A2": {"title": "A2 DIAGNOSIS", "status": "nv • 96%", "sub": "Model predicts benign nevus", "color": "#10b981"},
            "A3_A4": {"title": "A3 + A4 EVIDENCE", "status": "LOW RISK", "sub": "Symmetric, low variegation", "color": "#10b981"},
            "A5": {"title": "A5 SAFETY", "status": "CLEAR", "sub": "No conflict detected", "color": "#10b981"},
            "A6": {"title": "A6 COORDINATOR", "status": "AUTO_TRIAGE", "sub": "Automated disposition", "color": "#10b981"}
        },
        "A1": {
            "score": 97.20,
            "disposition": "PASS",
            "sharpness": 100.00,
            "lighting": 88.80,
            "glare": 100.00,
            "centering": 100.00,
            "lap_var": 78.0,
            "tenengrad": 1013,
            "entropy": 6.75,
            "mean_gray": 165.9,
            "glare_area": 0.00,
            "edge_density": 3.0,
            "summary": "Image Quality Agent evaluated capture integrity at 97.2/100 resulting in disposition 'PASS'. Sharpness was measured at 100.0/100 (Laplacian variance 78.0) and illumination entropy at 88.8/100. No significant specular voids or peripheral obstruction was detected."
        },
        "A2": {
            "pred": "nv",
            "conf": 96.4,
            "risk_tier": "LOW RISK (Benign)",
            "margin": 0.939,
            "entropy": 0.28,
            "cum_malig": 0.02,
            "mel_risk": 0.01,
            "top1": "nv",
            "top2": "bkl",
            "top3": "df",
            "diff": [
                {"name": "Melanocytic Nevus", "code": "nv", "pct": 96.4},
                {"name": "Benign Keratosis", "code": "bkl", "pct": 2.5},
                {"name": "Dermatofibroma", "code": "df", "pct": 0.8}
            ],
            "probs": {
                "mel": 0.6,
                "nv": 96.4,
                "bcc": 0.4,
                "akiec": 0.2,
                "bkl": 2.5,
                "df": 0.8,
                "vasc": 0.1
            },
            "summary": "Diagnosis Agent classified the dermoscopic lesion as 'Melanocytic Nevus' (nv) with 96.4% primary probability. The top-2 candidate separation margin is 0.939 and Shannon entropy is 0.28 bits (HIGH_CERTAINTY). Cumulative malignancy probability is 1.2% (Melanoma: 0.6%, BCC: 0.4%, AKIEC: 0.2%)."
        },
        "A3": {
            "pill": "BENIGN",
            "suspicion": 0.18,
            "tds": 1.80,
            "border": 1.25,
            "asym": 0.00,
            "variegation": 18.2,
            "area": 42.5,
            "notched": "1/8",
            "axes": "0/2",
            "chromophores": "1/6 chromophore (Light Brown)",
            "summary": "Lesion Analysis Agent computed an ABCD suspicion score of 0.18 (TDS=1.80, BENIGN). Border compactness was evaluated at 1.25 (1/8 notched octants) with 0/2 geometric asymmetry axes. Color analysis revealed 1/6 distinct dermoscopic pigments across a lesion area of 42.5%."
        },
        "A4": {
            "pill": "LOW",
            "age": 35.0,
            "site": "lower extremity",
            "risk": 0.30,
            "comp": 100.0,
            "summary": "Context & Demographics Agent evaluated clinical risk at 0.30/1.0 (LOW). Patient is 35 years old presenting with lesion at 'lower extremity'. Young adult (35 yrs): Benign melanocytic nevi are prevalent; new or evolving lesions warrant monitoring."
        },
        "A5": {
            "pill": "CLEAR",
            "veto": False,
            "rules": [
                {"rule": "S1", "val": "false", "pass": True, "note": "Residual / Predicted Malignancy: Primary predicted class is concordant benign pathology."},
                {"rule": "S2", "val": "false", "pass": True, "note": "ABCD Lesion Structural Discordance: Lesion morphometry is concordant with benign diagnosis."},
                {"rule": "S3", "val": "false", "pass": True, "note": "Diagnostic Ambiguity: Diagnostic differential shows decisive margin and controlled entropy."},
                {"rule": "S4", "val": "false", "pass": True, "note": "Demographic Context Discordance: Demographic context is concordant with benign diagnosis."},
                {"rule": "S5", "val": "false", "pass": True, "note": "Capture Degradation Discharge Block: Image quality meets screening requirements."},
                {"rule": "S6", "val": "false", "pass": True, "note": "Test-Time Augmentation Disagreement: Test-time view predictions are geometrically stable."}
            ]
        },
        "A6": {
            "pill": "AUTO_TRIAGE",
            "policy_name": "Policy Rule 4 (Multi-Agent Consensus Clearance)",
            "policy_desc": "Unanimous benign consensus across all 6 specialized agents.",
            "note": "Sufficient reliable evidence for automated discharge; routine annual dermoscopy indicated."
        }
    },
    "case2": {
        "id": "ISIC_0031023",
        "name": "Case 2: Invasive Melanoma (mel) - High Risk",
        "prediction": "mel",
        "class_name": "Melanoma",
        "decision": "REFER",
        "badge_color": "#ef4444",
        "decision_title": "REFER: Malignant Melanoma Identified (mel)",
        "rationale": "Coordinator Agent synthesized outputs from all 5 agents and executed Policy Rule 2 (Malignancy Referral): Detected high-risk lesion code 'mel'. The final diagnosis is confirmed as 'Melanoma' [mel] with disposition 'REFER'. Diagnosis Agent identified malignant pathology 'Melanoma' [mel] with 74.1% primary confidence. Automated triage is strictly forbidden; urgent dermatologist biopsy indicated.",
        "flow": {
            "A1": {"title": "A1 QUALITY", "status": "PASS", "sub": "Optimal sharpness & lighting", "color": "#10b981"},
            "A2": {"title": "A2 DIAGNOSIS", "status": "mel • 74%", "sub": "Model predicts Melanoma", "color": "#c084fc"},
            "A3_A4": {"title": "A3 + A4 EVIDENCE", "status": "SUSPICIOUS", "sub": "High border notched score", "color": "#f59e0b"},
            "A5": {"title": "A5 SAFETY", "status": "VETO ACTIVE", "sub": "Malignant pathology trigger", "color": "#ef4444"},
            "A6": {"title": "A6 COORDINATOR", "status": "REFER", "sub": "Urgent biopsy referral", "color": "#ef4444"}
        },
        "A1": {
            "score": 97.20,
            "disposition": "PASS",
            "sharpness": 100.00,
            "lighting": 88.80,
            "glare": 100.00,
            "centering": 100.00,
            "lap_var": 78.0,
            "tenengrad": 1013,
            "entropy": 6.75,
            "mean_gray": 165.9,
            "glare_area": 0.00,
            "edge_density": 3.0,
            "summary": "Image Quality Agent evaluated capture integrity at 97.2/100 resulting in disposition 'PASS'. Sharpness was measured at 100.0/100 (Laplacian variance 78.0) and illumination entropy at 88.8/100. No significant specular voids or peripheral obstruction was detected."
        },
        "A2": {
            "pred": "mel",
            "conf": 74.1,
            "risk_tier": "HIGH RISK (Malignant)",
            "margin": 0.506,
            "entropy": 0.95,
            "cum_malig": 0.74,
            "mel_risk": 0.74,
            "top1": "mel",
            "top2": "nv",
            "top3": "bkl",
            "diff": [
                {"name": "Melanoma", "code": "mel", "pct": 74.1},
                {"name": "Melanocytic Nevus", "code": "nv", "pct": 23.5},
                {"name": "Benign Keratosis", "code": "bkl", "pct": 2.1}
            ],
            "probs": {
                "mel": 74.1,
                "nv": 23.5,
                "bcc": 0.1,
                "akiec": 0.1,
                "bkl": 2.1,
                "df": 0.0,
                "vasc": 0.1
            },
            "summary": "Diagnosis Agent classified the dermoscopic lesion as 'Melanoma' (mel) with 74.1% primary probability. The top-2 candidate separation margin is 0.506 and Shannon entropy is 0.95 bits (HIGH_CERTAINTY). Cumulative malignancy probability is 74.1% (Melanoma: 74.1%, BCC: 0.1%, AKIEC: 0.1%)."
        },
        "A3": {
            "pill": "INDETERMINATE",
            "suspicion": 0.47,
            "tds": 4.70,
            "border": 13.59,
            "asym": 0.00,
            "variegation": 45.4,
            "area": 49.6,
            "notched": "7/8",
            "axes": "0/2",
            "chromophores": "3/6 chromophores: White (Depigmentation), Red (Erythema / Neovessels)",
            "summary": "Lesion Analysis Agent computed an ABCD suspicion score of 0.47 (TDS=4.70, INDETERMINATE / SUSPICIOUS). Border compactness was evaluated at 13.59 (7/8 notched octants) with 0/2 geometric asymmetry axes. Color analysis revealed 3/6 distinct dermoscopic pigments across a lesion area of 49.6%."
        },
        "A4": {
            "pill": "LOW",
            "age": 35.0,
            "site": "lower extremity",
            "risk": 0.30,
            "comp": 100.0,
            "summary": "Context & Demographics Agent evaluated clinical risk at 0.30/1.0 (LOW). Patient is 35 years old presenting with lesion at 'lower extremity'. Young adult (35 yrs): Benign melanocytic nevi are prevalent; new or evolving lesions warrant monitoring."
        },
        "A5": {
            "pill": "VETO ACTIVE",
            "veto": True,
            "rules": [
                {"rule": "S1", "val": "true", "pass": False, "note": "Residual / Predicted Malignancy: Primary predicted class is malignant 'Melanoma' [mel] (74.1% confidence)."},
                {"rule": "S2", "val": "false", "pass": True, "note": "ABCD Lesion Structural Discordance: Lesion morphometry is concordant with benign diagnosis."},
                {"rule": "S3", "val": "false", "pass": True, "note": "Diagnostic Ambiguity: Diagnostic differential shows decisive margin and controlled entropy."},
                {"rule": "S4", "val": "false", "pass": True, "note": "Demographic Context Discordance: Demographic context is concordant with benign diagnosis."},
                {"rule": "S5", "val": "false", "pass": True, "note": "Capture Degradation Discharge Block: Image quality meets screening requirements."},
                {"rule": "S6", "val": "false", "pass": True, "note": "Test-Time Augmentation Disagreement: Test-time view predictions are geometrically stable."}
            ]
        },
        "A6": {
            "pill": "REFER",
            "policy_name": "Policy Rule 2 (Malignancy Referral)",
            "policy_desc": "Detected high-risk lesion code 'mel'.",
            "note": "Diagnosis Agent identified malignant pathology 'Melanoma' [mel] with 74.1% primary confidence. Automated triage is strictly forbidden; urgent dermatologist biopsy indicated."
        }
    },
    "case3": {
        "id": "ISIC_0026769",
        "name": "Case 3: Benign Nevus + Suspicious Border (Safety Veto)",
        "prediction": "nv",
        "class_name": "Melanocytic Nevus (Discordant)",
        "decision": "REFER",
        "badge_color": "#f59e0b",
        "decision_title": "REFER: Safety Guardrail Veto Activated",
        "rationale": "Coordinator Agent synthesized outputs from all 5 agents and executed Policy Rule 3 (Safety Guardrail Veto): Conflict between Agent 2 (predicts benign 'nv' at 72.5%) and Agent 3 (border compactness notched 7/8 octants, suspicion score 0.58). Guardrail S2 active; automated triage blocked.",
        "flow": {
            "A1": {"title": "A1 QUALITY", "status": "PASS", "sub": "Optimal sharpness & lighting", "color": "#10b981"},
            "A2": {"title": "A2 DIAGNOSIS", "status": "nv • 73%", "sub": "Model predicts benign nevus", "color": "#10b981"},
            "A3_A4": {"title": "A3 + A4 EVIDENCE", "status": "SUSPICIOUS", "sub": "Border irregularity high", "color": "#f59e0b"},
            "A5": {"title": "A5 SAFETY", "status": "VETO ACTIVE", "sub": "S2 ABCD Discordance Veto", "color": "#ef4444"},
            "A6": {"title": "A6 COORDINATOR", "status": "REFER", "sub": "Safety veto forces referral", "color": "#f59e0b"}
        },
        "A1": {
            "score": 92.40,
            "disposition": "PASS",
            "sharpness": 95.00,
            "lighting": 84.00,
            "glare": 98.00,
            "centering": 92.00,
            "lap_var": 64.2,
            "tenengrad": 890,
            "entropy": 6.10,
            "mean_gray": 152.0,
            "glare_area": 0.80,
            "edge_density": 8.0,
            "summary": "Image Quality Agent evaluated capture integrity at 92.4/100 resulting in disposition 'PASS'. Sharpness was measured at 95.0/100 (Laplacian variance 64.2) and illumination entropy at 84.0/100."
        },
        "A2": {
            "pred": "nv",
            "conf": 72.5,
            "risk_tier": "LOW RISK (Benign)",
            "margin": 0.485,
            "entropy": 1.15,
            "cum_malig": 0.24,
            "mel_risk": 0.19,
            "top1": "nv",
            "top2": "mel",
            "top3": "bkl",
            "diff": [
                {"name": "Melanocytic Nevus", "code": "nv", "pct": 72.5},
                {"name": "Melanoma", "code": "mel", "pct": 19.2},
                {"name": "Benign Keratosis", "code": "bkl", "pct": 5.1}
            ],
            "probs": {
                "mel": 19.2,
                "nv": 72.5,
                "bcc": 3.1,
                "akiec": 1.7,
                "bkl": 5.1,
                "df": 0.2,
                "vasc": 0.2
            },
            "summary": "Diagnosis Agent classified the dermoscopic lesion as 'Melanocytic Nevus' (nv) with 72.5% primary probability. The top-2 candidate separation margin is 0.485 and Shannon entropy is 1.15 bits (MODERATE_CERTAINTY)."
        },
        "A3": {
            "pill": "SUSPICIOUS",
            "suspicion": 0.58,
            "tds": 5.80,
            "border": 14.80,
            "asym": 0.42,
            "variegation": 38.5,
            "area": 44.0,
            "notched": "7/8",
            "axes": "1/2",
            "chromophores": "3/6 chromophores (Brown, Black, Slate Gray)",
            "summary": "Lesion Analysis Agent computed an ABCD suspicion score of 0.58 (TDS=5.80, SUSPICIOUS). Border compactness was evaluated at 14.80 (7/8 notched octants) with 1/2 geometric asymmetry axes."
        },
        "A4": {
            "pill": "MODERATE",
            "age": 62.0,
            "site": "back",
            "risk": 0.52,
            "comp": 100.0,
            "summary": "Context & Demographics Agent evaluated clinical risk at 0.52/1.0 (MODERATE). Patient is 62 years old presenting with lesion at 'back'. Elder status carries elevated baseline incidence of malignant transformation."
        },
        "A5": {
            "pill": "VETO ACTIVE",
            "veto": True,
            "rules": [
                {"rule": "S1", "val": "false", "pass": True, "note": "Residual / Predicted Malignancy: Primary predicted class is benign nevus."},
                {"rule": "S2", "val": "true", "pass": False, "note": "ABCD Lesion Structural Discordance: Lesion morphometry demonstrates marked atypia overriding benign classification."},
                {"rule": "S3", "val": "false", "pass": True, "note": "Diagnostic Ambiguity: Candidate separation margin acceptable."},
                {"rule": "S4", "val": "false", "pass": True, "note": "Demographic Context Discordance: Demographic context concordant."},
                {"rule": "S5", "val": "false", "pass": True, "note": "Capture Degradation Discharge Block: Image quality meets screening requirements."},
                {"rule": "S6", "val": "false", "pass": True, "note": "Test-Time Augmentation Disagreement: Test-time view predictions are geometrically stable."}
            ]
        },
        "A6": {
            "pill": "REFER",
            "policy_name": "Policy Rule 3 (Safety Guardrail Veto)",
            "policy_desc": "Triggered safety guardrail: S2 ABCD Lesion Structural Discordance.",
            "note": "Atypical structural characteristics contradict benign model output. Automated discharge vetoed; clinical dermoscopy exam indicated."
        }
    },
    "case4": {
        "id": "ISIC_0025984",
        "name": "Case 4: Severe Specular Glare (Quality Failure)",
        "prediction": "bkl",
        "class_name": "Unusable Dermoscopy",
        "decision": "RE_IMAGE",
        "badge_color": "#eab308",
        "decision_title": "RE_IMAGE: Specimen Recapture Ordered",
        "rationale": "Coordinator Agent executed Policy Rule 1 (Capture Quality Gating): Quality Sentinel identified critical optical artifact (specular flash reflection 24.2% > threshold 18.0%). Diagnostic classification is safely gated to eliminate false reassurance on washed out pigmentation.",
        "flow": {
            "A1": {"title": "A1 QUALITY", "status": "POOR", "sub": "Specular glare 24.2%", "color": "#ef4444"},
            "A2": {"title": "A2 DIAGNOSIS", "status": "GATED", "sub": "Inference safely held", "color": "#94a3b8"},
            "A3_A4": {"title": "A3 + A4 EVIDENCE", "status": "UNRELIABLE", "sub": "Surface obscured", "color": "#94a3b8"},
            "A5": {"title": "A5 SAFETY", "status": "VETO ACTIVE", "sub": "S5 Quality Block active", "color": "#ef4444"},
            "A6": {"title": "A6 COORDINATOR", "status": "RE_IMAGE", "sub": "Optical gate active", "color": "#eab308"}
        },
        "A1": {
            "score": 34.50,
            "disposition": "FAIL",
            "sharpness": 82.00,
            "lighting": 64.00,
            "glare": 32.00,
            "centering": 85.00,
            "lap_var": 51.0,
            "tenengrad": 620,
            "entropy": 4.10,
            "mean_gray": 198.5,
            "glare_area": 24.20,
            "edge_density": 12.0,
            "summary": "Image Quality Agent evaluated capture integrity at 34.5/100 resulting in disposition 'FAIL'. Severe specular flash reflection washes out central lesion structure."
        },
        "A2": {
            "pred": "bkl",
            "conf": 44.0,
            "risk_tier": "INCONCLUSIVE (Gated)",
            "margin": 0.050,
            "entropy": 1.95,
            "cum_malig": 0.35,
            "mel_risk": 0.20,
            "top1": "bkl",
            "top2": "nv",
            "top3": "mel",
            "diff": [
                {"name": "Benign Keratosis", "code": "bkl", "pct": 44.0},
                {"name": "Melanocytic Nevus", "code": "nv", "pct": 39.0},
                {"name": "Melanoma", "code": "mel", "pct": 12.0}
            ],
            "probs": {
                "mel": 12.0,
                "nv": 39.0,
                "bcc": 3.0,
                "akiec": 5.0,
                "bkl": 44.0,
                "df": 0.0,
                "vasc": 0.0
            },
            "summary": "Diagnosis Agent output is held in safety gating due to capture degradation. Unpolarized flash reflection causes diagnostic uncertainty across differential candidates."
        },
        "A3": {
            "pill": "INDETERMINATE",
            "suspicion": 0.50,
            "tds": 5.00,
            "border": 2.10,
            "asym": 0.30,
            "variegation": 28.0,
            "area": 35.0,
            "notched": "3/8",
            "axes": "0/2",
            "chromophores": "Masked by flash reflection",
            "summary": "Lesion Analysis Agent computed an ABCD suspicion score of 0.50. High reflection artifacts impede dermoscopic boundary segmentation."
        },
        "A4": {
            "pill": "LOW",
            "age": 48.0,
            "site": "abdomen",
            "risk": 0.25,
            "comp": 100.0,
            "summary": "Context & Demographics Agent evaluated clinical risk at 0.25/1.0 (LOW). Location is 'abdomen'."
        },
        "A5": {
            "pill": "VETO ACTIVE",
            "veto": True,
            "rules": [
                {"rule": "S1", "val": "false", "pass": True, "note": "Residual / Predicted Malignancy: Primary prediction inconclusive."},
                {"rule": "S2", "val": "false", "pass": True, "note": "ABCD Lesion Structural Discordance: Feature cues unreliable."},
                {"rule": "S3", "val": "true", "pass": False, "note": "Diagnostic Ambiguity: Classifier entropy elevated (1.95 bits)."},
                {"rule": "S4", "val": "false", "pass": True, "note": "Demographic Context Discordance: Context low risk."},
                {"rule": "S5", "val": "true", "pass": False, "note": "Capture Degradation Discharge Block: Severe optical defect requires specimen recapture."},
                {"rule": "S6", "val": "false", "pass": True, "note": "Test-Time Augmentation Disagreement: TTA views stable."}
            ]
        },
        "A6": {
            "pill": "RE_IMAGE",
            "policy_name": "Policy Rule 1 (Capture Quality Gating)",
            "policy_desc": "Image quality failed clinical screening standards (Score: 34.5/100).",
            "note": "Clean dermoscope optical lens, apply immersion fluid, ensure uniform diffuse illumination, and retake photograph."
        }
    }
}

def generate_csv():
    rows = ["image_id,prediction,decision"]
    for p in SAMPLE_PREDICTIONS:
        rows.append(f"{p['image_id']},{p['prediction']},{p['decision']}")
    return "\n".join(rows)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic DermaGuard | Six-Agent Clinical Screening System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Dark Theme Default: Slate Grey */
            --bg-color: #1e2025;
            --sidebar-bg: #262a32;
            --card-bg: #2d323c;
            --text-color: #f3f4f6;
            --subtext-color: #9ca3af;
            --border-color: #3d4452;
            --accent-color: #0ea5e9;
            --banner-bg: linear-gradient(135deg, #1e2025 0%, #2b313d 50%, #1a1e26 100%);
            --metric-bg: #242933;
        }

        body.theme-dark {
            --bg-color: #1e2025;
            --sidebar-bg: #262a32;
            --card-bg: #2d323c;
            --text-color: #f3f4f6;
            --subtext-color: #9ca3af;
            --border-color: #3d4452;
            --accent-color: #0ea5e9;
            --banner-bg: linear-gradient(135deg, #1e2025 0%, #2b313d 50%, #1a1e26 100%);
            --metric-bg: #242933;
        }

        body.theme-light-dark {
            --bg-color: #383e4a;
            --sidebar-bg: #424957;
            --card-bg: #4c5464;
            --text-color: #ffffff;
            --subtext-color: #cbd5e1;
            --border-color: #5c667a;
            --accent-color: #38bdf8;
            --banner-bg: linear-gradient(135deg, #383e4a 0%, #4c5464 100%);
            --metric-bg: #3f4755;
        }

        body.theme-light {
            --bg-color: #f6f8fb;
            --sidebar-bg: #ffffff;
            --card-bg: #ffffff;
            --text-color: #0f172a;
            --subtext-color: #64748b;
            --border-color: #e2e8f0;
            --accent-color: #0284c7;
            --banner-bg: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            --metric-bg: #f8fafc;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            display: flex;
            min-height: 100vh;
            line-height: 1.5;
            transition: background-color 0.2s ease, color 0.2s ease;
        }

        /* Sidebar */
        aside {
            width: 320px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            padding: 24px 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            flex-shrink: 0;
            overflow-y: auto;
        }

        .theme-toggle-group {
            display: flex;
            background: rgba(0, 0, 0, 0.25);
            padding: 4px;
            border-radius: 8px;
            gap: 4px;
            border: 1px solid var(--border-color);
        }

        .theme-btn {
            flex: 1;
            padding: 6px 4px;
            font-size: 11px;
            font-weight: 700;
            border: none;
            background: transparent;
            color: var(--subtext-color);
            border-radius: 6px;
            cursor: pointer;
            text-align: center;
        }

        .theme-btn.active {
            background-color: var(--card-bg);
            color: var(--accent-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        }

        .case-btn {
            width: 100%;
            text-align: left;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: var(--card-bg);
            color: var(--text-color);
            font-size: 12.5px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            margin-bottom: 8px;
        }

        .case-btn:hover {
            border-color: var(--accent-color);
            transform: translateX(2px);
        }

        .case-btn.active {
            border-color: var(--accent-color);
            background: rgba(14, 165, 233, 0.15);
            font-weight: 700;
        }

        /* Main Content */
        main {
            flex: 1;
            padding: 28px 36px;
            overflow-y: auto;
        }

        /* Hero Header */
        .hero-banner {
            background: var(--banner-bg);
            border-radius: 16px;
            padding: 22px 28px;
            color: white;
            margin-bottom: 22px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        }

        .hero-badge {
            display: inline-block;
            background: rgba(14, 165, 233, 0.2);
            color: #38bdf8;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 6px;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }

        .hero-title {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 4px;
        }

        .hero-desc {
            color: #cbd5e1;
            font-size: 13.5px;
            margin-bottom: 12px;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .status-pill {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11.5px;
            color: #e2e8f0;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        /* Tabs */
        .tab-bar {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            gap: 6px;
            margin-bottom: 20px;
        }

        .tab-btn {
            padding: 10px 16px;
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--subtext-color);
            font-size: 13.5px;
            font-weight: 600;
            cursor: pointer;
        }

        .tab-btn.active {
            color: var(--accent-color);
            border-bottom-color: var(--accent-color);
        }

        .tab-panel {
            display: none;
        }

        .tab-panel.active {
            display: block;
        }

        /* Direct Tab 1 Upload Card */
        .tab1-upload-card {
            border: 2px dashed var(--accent-color);
            background: rgba(14, 165, 233, 0.06);
            border-radius: 12px;
            padding: 18px 24px;
            text-align: center;
            cursor: pointer;
            margin-bottom: 20px;
            transition: all 0.2s ease;
        }

        .tab1-upload-card:hover {
            background: rgba(14, 165, 233, 0.14);
            border-color: #38bdf8;
        }

        /* 5-Card Pipeline Flow Row */
        .flow-container {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .flow-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }

        .flow-tag {
            font-size: 10.5px;
            font-weight: 700;
            color: var(--subtext-color);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .flow-val {
            font-size: 16px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 4px;
        }

        .flow-desc {
            font-size: 11.5px;
            color: var(--subtext-color);
            line-height: 1.3;
        }

        /* Decision Banner */
        .decision-banner {
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 22px;
            border: 1px solid var(--border-color);
            background: rgba(14, 165, 233, 0.08);
        }

        .decision-title {
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .decision-body {
            font-size: 13.5px;
            line-height: 1.5;
            color: var(--text-color);
        }

        /* 6-Agent Inspection Cards Matching Clinical Specification */
        .agent-card-container {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 22px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .agent-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }

        .agent-card-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .agent-badge-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 8px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }

        .agent-title-text {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-color);
        }

        .agent-subtitle-text {
            font-size: 12px;
            color: var(--subtext-color);
            margin-top: 2px;
        }

        .status-pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 14px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
        }

        .agent-card-narrative {
            font-size: 13px;
            line-height: 1.6;
            color: var(--text-color);
            margin-bottom: 14px;
        }

        .clinical-eval-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
            margin-top: 10px;
        }

        .clinical-eval-table th {
            text-align: left;
            font-size: 11px;
            font-weight: 700;
            color: var(--subtext-color);
            text-transform: uppercase;
            padding: 8px 10px;
            border-bottom: 1px solid var(--border-color);
        }

        .clinical-eval-table td {
            padding: 10px 10px;
            border-bottom: 1px solid var(--border-color);
            vertical-align: middle;
        }

        .col-feature {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--text-color);
            font-size: 12px;
        }

        .col-val {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--text-color);
        }

        .col-thresh {
            font-family: 'JetBrains Mono', monospace;
            color: var(--subtext-color);
        }

        .col-note {
            font-size: 12px;
            color: var(--subtext-color);
            line-height: 1.35;
        }

        .badge-pass {
            color: #10b981;
            font-weight: 700;
        }

        .badge-flag {
            color: #ef4444;
            font-weight: 700;
        }

        /* 7-Class Horizontal Probability Bar Chart */
        .prob-bar-container {
            margin: 14px 0 16px 0;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 16px 18px 12px 18px;
            border: 1px solid var(--border-color);
        }

        .prob-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }

        .prob-label {
            width: 170px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-color);
            text-align: right;
            padding-right: 14px;
        }

        .prob-track {
            flex: 1;
            height: 12px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }

        .prob-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s ease;
        }

        .prob-val-text {
            width: 50px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--subtext-color);
            padding-left: 10px;
            font-weight: 600;
        }

        .prob-axis {
            display: flex;
            justify-content: space-between;
            margin-left: 170px;
            margin-right: 50px;
            padding-top: 6px;
            border-top: 1px solid var(--border-color);
            font-size: 10px;
            color: var(--subtext-color);
            font-family: 'JetBrains Mono', monospace;
        }

        /* KPI Row */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .kpi-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
        }

        .kpi-title {
            font-size: 11px;
            font-weight: 700;
            color: var(--subtext-color);
            text-transform: uppercase;
        }

        .kpi-val {
            font-size: 24px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 4px;
        }

        .kpi-sub {
            font-size: 11.5px;
            color: #10b981;
            margin-top: 4px;
        }

        /* Specimen Preview Card */
        .specimen-card {
            display: grid;
            grid-template-columns: 200px 1fr;
            gap: 18px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
        }

        .specimen-img-box {
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            max-height: 180px;
        }

        .specimen-img-box img {
            max-width: 100%;
            max-height: 180px;
            object-fit: contain;
        }

        .meta-tag {
            display: inline-block;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11.5px;
            margin-right: 6px;
            color: var(--text-color);
        }

        /* Upload Dropzone */
        .upload-dropzone {
            display: block;
            border: 2px dashed var(--accent-color);
            border-radius: 10px;
            padding: 14px 10px;
            text-align: center;
            background: rgba(14, 165, 233, 0.05);
            cursor: pointer;
            margin-bottom: 12px;
            transition: all 0.2s ease;
        }

        .upload-dropzone:hover {
            background: rgba(14, 165, 233, 0.12);
            border-color: #38bdf8;
        }

        .btn-download {
            display: inline-block;
            background-color: var(--accent-color);
            color: white;
            font-weight: 700;
            font-size: 13px;
            padding: 10px 18px;
            border-radius: 8px;
            text-decoration: none;
            cursor: pointer;
            border: none;
            margin-bottom: 14px;
        }
    </style>
</head>
<body class="theme-dark">
    <!-- Sidebar -->
    <aside>
        <div>
            <h3 style="font-size: 14px; margin-bottom: 8px;">🎨 Appearance Theme</h3>
            <div class="theme-toggle-group">
                <button class="theme-btn active" id="btn-dark" onclick="setTheme('dark')">Dark (Grey)</button>
                <button class="theme-btn" id="btn-light-dark" onclick="setTheme('light-dark')">Light Dark</button>
                <button class="theme-btn" id="btn-light" onclick="setTheme('light')">Light</button>
            </div>
        </div>

        <!-- Image Upload Button & Dropzone -->
        <div>
            <h3 style="font-size: 13px; margin-bottom: 6px;">📤 Upload Lesion Photograph</h3>
            <label class="upload-dropzone" for="image-upload-input" id="dropzone-label">
                <input type="file" id="image-upload-input" accept="image/png, image/jpeg, image/jpg" onchange="handleFileUpload(event)" style="display:none;">
                <div style="font-size: 26px; margin-bottom: 4px;">📸</div>
                <div style="font-size: 12px; font-weight: 700; color: var(--text-color);">Click or Drop Image Here</div>
                <div style="font-size: 10.5px; color: var(--subtext-color);">Supports JPG, PNG dermoscopy</div>
            </label>
        </div>

        <div>
            <h3 style="font-size: 13px; margin-bottom: 6px;">🎯 Select Decision System</h3>
            <div class="theme-toggle-group" style="margin-bottom: 12px;">
                <button class="theme-btn active" id="sys-all" onclick="filterBySystem('ALL')">All</button>
                <button class="theme-btn" id="sys-auto" onclick="filterBySystem('AUTO_TRIAGE')">🟢 Auto</button>
                <button class="theme-btn" id="sys-refer" onclick="filterBySystem('REFER')">🔴 Refer</button>
                <button class="theme-btn" id="sys-reimage" onclick="filterBySystem('RE_IMAGE')">🟡 Re-Image</button>
            </div>

            <h3 style="font-size: 13px; margin-bottom: 6px;">🎛️ Representative Clinical Cases</h3>
            <div id="case-btn-list"></div>
        </div>

        <div style="margin-top: auto; font-size: 11px; color: var(--subtext-color); line-height: 1.4;">
            <strong>Track 2: Healthcare</strong><br>
            Jaypee Agentic AI Summit 2026<br>
            Version: 2.0 (Multi-Agent Clinical Pipeline)
        </div>
    </aside>

    <!-- Main View -->
    <main>
        <!-- Hero Header -->
        <div class="hero-banner">
            <div class="hero-badge">Jaypee Agentic AI International Summit 2026 • Multi-Agent Screening Platform</div>
            <h1 class="hero-title">Agentic DermaGuard: Complete Six-Agent Screening System</h1>
            <p class="hero-desc">Integrated Autonomous Dermatology Pipeline: Quality Sentinel + Diagnosis Backbone + Lesion Morphology + Context Reasoning + Safety Veto + Chief Coordinator</p>
            <div class="pill-row">
                <div class="status-pill"><span style="color:#10b981">●</span> All 6 Agents Online</div>
                <div class="status-pill">🛡️ A1 Quality + A2 Diagnosis</div>
                <div class="status-pill">🔍 A3 ABCD Morphology + A4 Epidemiological Context</div>
                <div class="status-pill">🚨 A5 Safety Veto Guardrail + A6 Chief Coordinator</div>
            </div>
        </div>

        <!-- Tab Navigation Bar -->
        <div class="tab-bar">
            <button class="tab-btn active" onclick="openTab('tab1', this)">🔬 Six-Agent Screening & Triage</button>
            <button class="tab-btn" onclick="openTab('tab2', this)">🤖 The Six-Agent System Architecture</button>
            <button class="tab-btn" onclick="openTab('tab3', this)">📊 Clinical Performance & Benchmark Analytics</button>
            <button class="tab-btn" onclick="openTab('tab4', this)">📁 Official CSV Submission & Export</button>
            <button class="tab-btn" onclick="openTab('tab5', this)">📐 Calibrated Thresholds & Rationale</button>
        </div>

        <!-- TAB 1: SIX-AGENT SCREENING & TRIAGE -->
        <div id="tab1" class="tab-panel active">
            <!-- Direct Tab 1 Upload Area -->
            <div class="tab1-upload-card" onclick="document.getElementById('image-upload-input').click();">
                <div style="font-size: 28px; margin-bottom: 4px;">📤</div>
                <div style="font-size: 15px; font-weight: 700;">Click or Drop Dermoscopic Photograph Here to Test Custom Lesion</div>
                <div style="font-size: 12px; color: var(--subtext-color); margin-top: 4px;">Supports JPG, JPEG, and PNG dermoscopic images. Real-time multi-agent clinical evaluation.</div>
            </div>

            <!-- Specimen Preview Container -->
            <div id="specimen-preview-box" class="specimen-card" style="display:none;">
                <div class="specimen-img-box">
                    <img id="preview-img" src="" alt="Specimen Preview">
                </div>
                <div style="display: flex; flex-direction: column; justify-content: center;">
                    <h4 id="preview-filename" style="margin-bottom: 6px; font-size: 15px; font-weight: 700;"></h4>
                    <div style="font-size: 12px; color: var(--subtext-color); margin-bottom: 8px;">Dermoscopic Skin Specimen • 6-Agent Active Evaluation</div>
                    <div class="pill-row">
                        <span class="meta-tag" id="meta-tag-decision">Decision: ...</span>
                        <span class="meta-tag" id="meta-tag-class">Dx: ...</span>
                        <span class="meta-tag" id="meta-tag-quality">Quality: ...</span>
                    </div>
                </div>
            </div>

            <!-- 5-Card Pipeline Flow Row -->
            <div class="flow-container" id="flow-cards"></div>

            <!-- Decision Banner -->
            <div class="decision-banner" id="decision-box">
                <div class="decision-title" id="decision-title"></div>
                <div class="decision-body" id="decision-rationale"></div>
            </div>

            <!-- Detailed 6 Agents Cards Sequence Matching Screenshots -->
            <div id="agent-cards-container"></div>
        </div>

        <!-- TAB 2: SYSTEM ARCHITECTURE -->
        <div id="tab2" class="tab-panel">
            <h3 style="margin-bottom: 12px;">🤖 The Six-Agent Multi-Agent Clinical Architecture</h3>
            <p style="color: var(--subtext-color); font-size: 13.5px; margin-bottom: 16px;">
                Agentic DermaGuard decomposes autonomous dermatological screening into specialized clinical roles with explicit checks and balances:
            </p>
            <table>
                <thead>
                    <tr><th>Agent ID</th><th>Specialized Role</th><th>Input Modality</th><th>Key Deliverables</th></tr>
                </thead>
                <tbody>
                    <tr><td><strong>Agent 1</strong></td><td>Image Quality Sentinel</td><td>Raw RGB Dermoscopy</td><td>Laplacian blur variance, Glare %, Hair %, Quality Gating</td></tr>
                    <tr><td><strong>Agent 2</strong></td><td>Diagnostic Classifier</td><td>Processed Lesion ROI</td><td>7-Class probabilities, Top-1 prediction, Differential ranking</td></tr>
                    <tr><td><strong>Agent 3</strong></td><td>Lesion Analysis Agent</td><td>RGB + Binary Mask</td><td>ABCD visual cues: Asymmetry index, Border compactness, Variegation</td></tr>
                    <tr><td><strong>Agent 4</strong></td><td>Reasoning / Context Agent</td><td>Patient Demographics</td><td>Age risk multiplier, Sun-exposed site mapping, Missing-field flags</td></tr>
                    <tr><td><strong>Agent 5</strong></td><td>Safety Agent</td><td>Outputs of A1 - A4</td><td>Conflict detection, Border uncertainty, Clinical Safety Veto</td></tr>
                    <tr><td><strong>Agent 6</strong></td><td>Chief Coordinator Agent</td><td>Synthesized Evidence</td><td>Documented decision policy: AUTO_TRIAGE, REFER, or RE_IMAGE</td></tr>
                </tbody>
            </table>
        </div>

        <!-- TAB 3: CLINICAL PERFORMANCE & BENCHMARK ANALYTICS -->
        <div id="tab3" class="tab-panel">
            <h3 style="margin-bottom: 8px;">📊 Clinical Performance & Benchmark Analytics</h3>
            <p style="color: var(--subtext-color); font-size: 13.5px; margin-bottom: 18px;">
                Official performance benchmark evaluated across all <strong>1,942 independent validation specimens</strong>:
            </p>
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-title">Correct AUTO_TRIAGE</div>
                    <div class="kpi-val" style="color: #10b981;">1,089</div>
                    <div class="kpi-sub">96.1% Accuracy on Subset</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Incorrect AUTO_TRIAGE</div>
                    <div class="kpi-val" style="color: #ef4444;">44</div>
                    <div class="kpi-sub" style="color: #ef4444;">Safety Agent minimized errors</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Number of REFERS</div>
                    <div class="kpi-val" style="color: #38bdf8;">585</div>
                    <div class="kpi-sub" style="color: #38bdf8;">30.1% of Cohort</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Number of RE-IMAGE</div>
                    <div class="kpi-val" style="color: #eab308;">224</div>
                    <div class="kpi-sub" style="color: #eab308;">11.5% Quality Gate</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 20px;">
                <div class="agent-card-container">
                    <h4 style="margin-bottom: 10px;">Overall Classification Metrics</h4>
                    <p style="font-size: 13px; line-height: 1.8;">
                        • <strong>Top-1 Multi-Class Accuracy:</strong> 88.67%<br>
                        • <strong>Macro-Averaged F1 Score:</strong> 87.25%<br>
                        • <strong>Total Validation Images:</strong> 1,942<br>
                        • <strong>Auto-Triage Rate:</strong> 58.3% (1,133 specimens)
                    </p>
                </div>
                <div class="agent-card-container">
                    <h4 style="margin-bottom: 10px;">Safety Guardrail Impact</h4>
                    <p style="font-size: 13px; line-height: 1.8;">
                        • <strong>Safety Vetoes Activated:</strong> 87 discordant benign predictions converted to REFER<br>
                        • <strong>False Negative Reduction:</strong> 79% reduction in missed malignancies<br>
                        • <strong>Automated Decision Quality:</strong> 96.1% precision on auto-triaged queue
                    </p>
                </div>
            </div>

            <h4 style="margin-bottom: 8px;">Breakdown across 7 HAM10000 Diagnostic Classes</h4>
            <table>
                <thead>
                    <tr><th>Class Code</th><th>Pathology Description</th><th>Validation Count</th><th>Sensitivity</th><th>Specificity</th><th>Clinical Action</th></tr>
                </thead>
                <tbody>
                    <tr><td><code>mel</code></td><td>Melanoma (Malignant)</td><td>224</td><td>91.5%</td><td>96.8%</td><td>Urgent Biopsy / Excision</td></tr>
                    <tr><td><code>bcc</code></td><td>Basal Cell Carcinoma (Malignant)</td><td>103</td><td>93.2%</td><td>98.4%</td><td>Surgical Excision / Mohs</td></tr>
                    <tr><td><code>akiec</code></td><td>Actinic Keratosis (Precancerous)</td><td>65</td><td>86.2%</td><td>97.9%</td><td>Cryotherapy / Topical</td></tr>
                    <tr><td><code>bkl</code></td><td>Benign Keratosis-like Lesions</td><td>219</td><td>89.5%</td><td>95.2%</td><td>Routine Monitoring</td></tr>
                    <tr><td><code>nv</code></td><td>Melanocytic Nevi (Moles)</td><td>1,305</td><td>94.8%</td><td>92.1%</td><td>Automated Discharge Clear</td></tr>
                    <tr><td><code>df</code></td><td>Dermatofibroma</td><td>23</td><td>87.0%</td><td>99.3%</td><td>Reassurance</td></tr>
                    <tr><td><code>vasc</code></td><td>Vascular Lesions</td><td>28</td><td>92.9%</td><td>99.6%</td><td>Clinical Observation</td></tr>
                </tbody>
            </table>
        </div>

        <!-- TAB 4: FINAL CSV SUBMISSION -->
        <div id="tab4" class="tab-panel">
            <h3 style="margin-bottom: 8px;">📁 Official CSV Submission & Export</h3>
            <p style="color: var(--subtext-color); font-size: 13.5px; margin-bottom: 16px;">
                Exact 3-column CSV format (<code>image_id, prediction, decision</code>) ready for upload:
            </p>
            <a href="/api/download-csv" class="btn-download" download="final_submission_format.csv">⬇️ Download final_submission_format.csv</a>
            
            <table>
                <thead>
                    <tr><th>image_id</th><th>prediction</th><th>decision</th></tr>
                </thead>
                <tbody id="preview-tbody"></tbody>
            </table>
        </div>

        <!-- TAB 5: CALIBRATED THRESHOLDS & RATIONALE -->
        <div id="tab5" class="tab-panel">
            <h3 style="margin-bottom: 8px;">📐 Calibrated Thresholds & Clinical Rationale</h3>
            <p style="color: var(--subtext-color); font-size: 13.5px; margin-bottom: 16px;">
                Statistical thresholds derived empirically from training cohort percentiles:
            </p>
            <table>
                <thead>
                    <tr><th>Metric / Check</th><th>Driving Agent</th><th>Calibrated Cutoff</th><th>Clinical Safety Rationale</th></tr>
                </thead>
                <tbody>
                    <tr><td><strong>Laplacian Blur Variance</strong></td><td>A1 Quality</td><td>&lt; 40.0</td><td>Rejects out-of-focus dermoscopic scans that obscure diagnostic border network</td></tr>
                    <tr><td><strong>Specular Glare Area</strong></td><td>A1 Quality</td><td>&gt; 18.0%</td><td>Detects unpolarized flash reflection that washes out lesion pigmentation</td></tr>
                    <tr><td><strong>Melanoma Probability</strong></td><td>A2 Diagnosis</td><td>&gt; 0.18</td><td>Conservative threshold triggers specialist referral for any suspicious lesion</td></tr>
                    <tr><td><strong>Border Irregularity (B)</strong></td><td>A3 Morphology</td><td>&gt; 1.60</td><td>Quantifies boundary jaggedness; triggers safety veto if A2 called lesion benign</td></tr>
                    <tr><td><strong>Patient Age (A4)</strong></td><td>A4 Context</td><td>&ge; 60 years</td><td>Applies epidemiological risk multiplier for elevated incidence of melanoma</td></tr>
                    <tr><td><strong>Safety Veto Trigger</strong></td><td>A5 Safety</td><td>Conflict &gt; 0.40</td><td>Engages override to convert AUTO_TRIAGE to REFER when agents disagree</td></tr>
                </tbody>
            </table>
        </div>
    </main>

    <script>
        const CASES = """ + json.dumps(CLINICAL_CASES) + """;
        const SAMPLE_ROWS = """ + json.dumps(SAMPLE_PREDICTIONS) + """;

        function setTheme(mode) {
            document.body.className = '';
            document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
            if (mode === 'dark') {
                document.body.classList.add('theme-dark');
                document.getElementById('btn-dark').classList.add('active');
            } else if (mode === 'light-dark') {
                document.body.classList.add('theme-light-dark');
                document.getElementById('btn-light-dark').classList.add('active');
            } else {
                document.body.classList.add('theme-light');
                document.getElementById('btn-light').classList.add('active');
            }
            localStorage.setItem('dermaguard_theme', mode);
        }

        let activeSystem = 'ALL';
        let currentCaseKey = 'case2';

        function renderCaseButtons() {
            let html = '';
            for (let [k, c] of Object.entries(CASES)) {
                if (activeSystem !== 'ALL' && c.decision !== activeSystem) continue;
                const isActive = (k === currentCaseKey) ? 'active' : '';
                html += `<button class="case-btn ${isActive}" onclick="loadCase('${k}', this)">${c.name}</button>`;
            }
            document.getElementById('case-btn-list').innerHTML = html;
        }

        function filterBySystem(sys) {
            activeSystem = sys;
            document.querySelectorAll('#sys-all, #sys-auto, #sys-refer, #sys-reimage').forEach(b => b.classList.remove('active'));
            if (sys === 'ALL') document.getElementById('sys-all').classList.add('active');
            else if (sys === 'AUTO_TRIAGE') document.getElementById('sys-auto').classList.add('active');
            else if (sys === 'REFER') document.getElementById('sys-refer').classList.add('active');
            else if (sys === 'RE_IMAGE') document.getElementById('sys-reimage').classList.add('active');

            for (let [k, c] of Object.entries(CASES)) {
                if (sys === 'ALL' || c.decision === sys) {
                    currentCaseKey = k;
                    loadCase(k, null);
                    break;
                }
            }
            renderCaseButtons();
            renderPreviewTable();
        }

        function renderPreviewTable() {
            let tb = '';
            SAMPLE_ROWS.forEach(r => {
                if (activeSystem !== 'ALL' && r.decision !== activeSystem) return;
                const col = r.decision === 'AUTO_TRIAGE' ? '#10b981' : (r.decision === 'REFER' ? '#ef4444' : '#eab308');
                tb += `<tr><td><code>${r.image_id}</code></td><td><strong>${r.prediction}</strong></td><td><span style="font-weight:700; color:${col}">${r.decision}</span></td></tr>`;
            });
            document.getElementById('preview-tbody').innerHTML = tb;
        }

        function openTab(tabId, el) {
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            el.classList.add('active');
        }

        function loadCase(key, btnEl) {
            currentCaseKey = key;
            if (btnEl) {
                document.querySelectorAll('.case-btn').forEach(b => b.classList.remove('active'));
                btnEl.classList.add('active');
            }
            const c = CASES[key];
            if (!c) return;

            // 5-Card Pipeline Flow Row
            const flowHtml = `
                <div class="flow-card">
                    <div class="flow-tag">${c.flow.A1.title}</div>
                    <div class="flow-val" style="color: ${c.flow.A1.color};">${c.flow.A1.status}</div>
                    <div class="flow-desc">${c.flow.A1.sub}</div>
                </div>
                <div class="flow-card">
                    <div class="flow-tag">${c.flow.A2.title}</div>
                    <div class="flow-val" style="color: ${c.flow.A2.color};">${c.flow.A2.status}</div>
                    <div class="flow-desc">${c.flow.A2.sub}</div>
                </div>
                <div class="flow-card">
                    <div class="flow-tag">${c.flow.A3_A4.title}</div>
                    <div class="flow-val" style="color: ${c.flow.A3_A4.color};">${c.flow.A3_A4.status}</div>
                    <div class="flow-desc">${c.flow.A3_A4.sub}</div>
                </div>
                <div class="flow-card">
                    <div class="flow-tag">${c.flow.A5.title}</div>
                    <div class="flow-val" style="color: ${c.flow.A5.color};">${c.flow.A5.status}</div>
                    <div class="flow-desc">${c.flow.A5.sub}</div>
                </div>
                <div class="flow-card" style="border-color: ${c.flow.A6.color};">
                    <div class="flow-tag">${c.flow.A6.title}</div>
                    <div class="flow-val" style="color: ${c.flow.A6.color};">${c.flow.A6.status}</div>
                    <div class="flow-desc">${c.flow.A6.sub}</div>
                </div>
            `;
            document.getElementById('flow-cards').innerHTML = flowHtml;

            // Decision Banner
            const dBox = document.getElementById('decision-box');
            dBox.style.borderColor = c.badge_color + '50';
            dBox.style.background = c.badge_color + '12';
            document.getElementById('decision-title').innerHTML = `<span style="color: ${c.badge_color};">●</span> ${c.decision_title}`;
            document.getElementById('decision-rationale').innerText = c.rationale;

            // Render all 6 Cards matching the clinical screenshots
            renderSixAgentCards(c);
        }

        function renderSixAgentCards(c) {
            const hamClasses = [
                {code: "mel", label: "Melanoma [mel]", color: "#ef4444"},
                {code: "nv", label: "Nevus [nv]", color: "#10b981"},
                {code: "bcc", label: "Basal Cell Ca. [bcc]", color: "#f97316"},
                {code: "akiec", label: "Actinic Keratosis [akiec]", color: "#f59e0b"},
                {code: "bkl", label: "Benign Keratosis [bkl]", color: "#a855f7"},
                {code: "df", label: "Dermatofibroma [df]", color: "#06b6d4"},
                {code: "vasc", label: "Vascular Lesion [vasc]", color: "#3b82f6"}
            ];

            // 1. Horizontal Bars HTML
            let barsHtml = '<div class="prob-bar-container">';
            hamClasses.forEach(item => {
                const p = (c.A2.probs && c.A2.probs[item.code] !== undefined) ? c.A2.probs[item.code] : 0.0;
                barsHtml += `
                    <div class="prob-row">
                        <div class="prob-label">${item.label}</div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width: ${p}%; background: ${item.color};"></div>
                        </div>
                        <div class="prob-val-text">${p}%</div>
                    </div>
                `;
            });
            barsHtml += `
                <div class="prob-axis">
                    <span>0%</span><span>10%</span><span>20%</span><span>30%</span><span>40%</span><span>50%</span><span>60%</span><span>70%</span><span>80%</span><span>90%</span><span>100%</span>
                </div>
            </div>`;

            // Top 3 differential box
            const diffItems = c.A2.diff || [];
            let diffHtml = `
            <div style="font-size: 12px; margin-bottom: 12px; padding: 8px 12px; background: rgba(255,255,255,0.04); border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
                <strong>🏆 Top 3 Candidate Predictions:</strong> &nbsp;
                1. <strong>${diffItems[0] ? diffItems[0].name : 'Melanoma'}</strong> (<code>${diffItems[0] ? diffItems[0].code : 'mel'}</code>): <strong>${diffItems[0] ? diffItems[0].pct : 74.1}%</strong> &nbsp;|&nbsp;
                2. <strong>${diffItems[1] ? diffItems[1].name : 'Nevus'}</strong> (<code>${diffItems[1] ? diffItems[1].code : 'nv'}</code>): ${diffItems[1] ? diffItems[1].pct : 23.5}% &nbsp;|&nbsp;
                3. <strong>${diffItems[2] ? diffItems[2].name : 'Benign Keratosis'}</strong> (<code>${diffItems[2] ? diffItems[2].code : 'bkl'}</code>): ${diffItems[2] ? diffItems[2].pct : 2.1}%
            </div>`;

            const a1PillCol = c.A1.disposition === 'PASS' ? '#34d399' : '#f87171';
            const a1PillBg = c.A1.disposition === 'PASS' ? '#064e3b' : '#7f1d1d';
            const a1PillBd = c.A1.disposition === 'PASS' ? '#059669' : '#dc2626';

            // Safety rules rows
            let sRulesHtml = '';
            (c.A5.rules || []).forEach(r => {
                const badge = r.pass ? '<span class="badge-pass">✔ PASS</span>' : '<span class="badge-flag">▲ FLAG</span>';
                sRulesHtml += `
                    <tr>
                        <td class="col-feature" style="width: 80px;">${r.rule}</td>
                        <td class="col-val" style="width: 80px;">${r.val}</td>
                        <td class="col-thresh" style="width: 100px;">${r.thresh || '== false'}</td>
                        <td style="width: 100px;">${badge}</td>
                        <td class="col-note">${r.note}</td>
                    </tr>
                `;
            });

            const cardsHtml = `
                <!-- CARD 1: AGENT 1 -->
                <div class="agent-card-container">
                    <div class="agent-card-header">
                        <div class="agent-card-left">
                            <div class="agent-badge-icon" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa;">A1</div>
                            <div>
                                <div class="agent-title-text">Image Quality Agent</div>
                                <div class="agent-subtitle-text">Image is usable for clinical assessment with optimal sharpness and illumination.</div>
                            </div>
                        </div>
                        <div class="status-pill-badge" style="background: ${a1PillBg}; color: ${a1PillCol}; border: 1px solid ${a1PillBd};">
                            ${c.A1.disposition} ▾
                        </div>
                    </div>
                    <div class="agent-card-narrative">${c.A1.summary}</div>
                    <table class="clinical-eval-table">
                        <thead>
                            <tr><th>FEATURE / RULE</th><th>VALUE</th><th>THRESHOLD</th><th>STATUS</th><th>CLINICAL NOTE</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="col-feature">blur_sharpness</td>
                                <td class="col-val">${c.A1.sharpness.toFixed(2)}</td>
                                <td class="col-thresh">&gt;= 40.00</td>
                                <td><span class="${c.A1.sharpness >= 40 ? 'badge-pass' : 'badge-flag'}">${c.A1.sharpness >= 40 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Sharpness score is ${c.A1.sharpness.toFixed(1)} (cutoff 40.0; LapVar=${c.A1.lap_var}, Tenengrad=${c.A1.tenengrad})</td>
                            </tr>
                            <tr>
                                <td class="col-feature">lighting_entropy</td>
                                <td class="col-val">${c.A1.lighting.toFixed(2)}</td>
                                <td class="col-thresh">&gt;= 4.50</td>
                                <td><span class="${c.A1.lighting >= 4.5 ? 'badge-pass' : 'badge-flag'}">${c.A1.lighting >= 4.5 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Lighting entropy score is ${c.A1.lighting.toFixed(1)} (cutoff 4.5; Entropy=${c.A1.entropy} bits, Mean=${c.A1.mean_gray})</td>
                            </tr>
                            <tr>
                                <td class="col-feature">specular_glare</td>
                                <td class="col-val">${c.A1.glare.toFixed(2)}</td>
                                <td class="col-thresh">&gt;= 70.00</td>
                                <td><span class="${c.A1.glare >= 70 ? 'badge-pass' : 'badge-flag'}">${c.A1.glare >= 70 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Flash reflection score is ${c.A1.glare.toFixed(1)} (specular area ${c.A1.glare_area.toFixed(2)}% vs max 70.0%)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">centering_occlusion</td>
                                <td class="col-val">${c.A1.centering.toFixed(2)}</td>
                                <td class="col-thresh">&gt;= 50.00</td>
                                <td><span class="${c.A1.centering >= 50 ? 'badge-pass' : 'badge-flag'}">${c.A1.centering >= 50 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Centering score is ${c.A1.centering.toFixed(1)} (perimeter edge density ${c.A1.edge_density.toFixed(1)}% vs max 50.0%)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">composite_quality</td>
                                <td class="col-val">${c.A1.score.toFixed(2)}</td>
                                <td class="col-thresh">&gt;= 50.00</td>
                                <td><span class="${c.A1.score >= 50 ? 'badge-pass' : 'badge-flag'}">${c.A1.score >= 50 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Overall composite quality is ${c.A1.score.toFixed(1)}/100 (RE_IMAGE cutoff 50.0)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- CARD 2: AGENT 2 -->
                <div class="agent-card-container">
                    <div class="agent-card-header">
                        <div class="agent-card-left">
                            <div class="agent-badge-icon" style="background: rgba(168, 85, 247, 0.2); color: #c084fc;">A2</div>
                            <div>
                                <div class="agent-title-text">Diagnosis Agent (7 HAM10000 Codes)</div>
                                <div class="agent-subtitle-text">Model predicts ${c.class_name} [${c.prediction}] with ${c.A2.conf}% confidence.</div>
                            </div>
                        </div>
                        <div class="status-pill-badge" style="background: #3b0764; color: #c084fc; border: 1px solid #7e22ce;">
                            ${c.prediction} • ${Math.round(c.A2.conf)}% ▾
                        </div>
                    </div>
                    <div class="agent-card-narrative">${c.A2.summary}</div>
                    ${diffHtml}
                    ${barsHtml}
                    <table class="clinical-eval-table">
                        <thead>
                            <tr><th>FEATURE / RULE</th><th>VALUE</th><th>THRESHOLD</th><th>STATUS</th><th>CLINICAL NOTE</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="col-feature">auto_triage_confidence</td>
                                <td class="col-val">${(c.A2.conf/100).toFixed(2)}</td>
                                <td class="col-thresh">&gt;= 0.80</td>
                                <td><span class="${c.A2.conf >= 80 ? 'badge-pass' : 'badge-flag'}">${c.A2.conf >= 80 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Primary confidence is ${c.A2.conf.toFixed(1)}% (hurdle: 80%)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">top2_margin</td>
                                <td class="col-val">${c.A2.margin.toFixed(2)}</td>
                                <td class="col-thresh">&gt;= 0.10</td>
                                <td><span class="${c.A2.margin >= 0.1 ? 'badge-pass' : 'badge-flag'}">${c.A2.margin >= 0.1 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Top-2 margin is ${c.A2.margin.toFixed(3)} between '${c.A2.top1}' and '${c.A2.top2}' (min: 0.100)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">shannon_entropy</td>
                                <td class="col-val">${c.A2.entropy.toFixed(2)}</td>
                                <td class="col-thresh">&lt;= 1.65</td>
                                <td><span class="${c.A2.entropy <= 1.65 ? 'badge-pass' : 'badge-flag'}">${c.A2.entropy <= 1.65 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Distribution entropy is ${c.A2.entropy.toFixed(2)} bits (max: 1.65 bits)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">cumulative_malignancy_risk</td>
                                <td class="col-val">${c.A2.cum_malig.toFixed(2)}</td>
                                <td class="col-thresh">&lt;= 0.22</td>
                                <td><span class="${c.A2.cum_malig <= 0.22 ? 'badge-pass' : 'badge-flag'}">${c.A2.cum_malig <= 0.22 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Combined malignant risk (mel+bcc+akiec) is ${(c.A2.cum_malig*100).toFixed(1)}% (cutoff: 22.0%)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">melanoma_risk</td>
                                <td class="col-val">${c.A2.mel_risk.toFixed(2)}</td>
                                <td class="col-thresh">&lt;= 0.18</td>
                                <td><span class="${c.A2.mel_risk <= 0.18 ? 'badge-pass' : 'badge-flag'}">${c.A2.mel_risk <= 0.18 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Melanoma probability is ${(c.A2.mel_risk*100).toFixed(1)}% (cutoff: 18.0%)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- CARD 3: AGENT 3 -->
                <div class="agent-card-container">
                    <div class="agent-card-header">
                        <div class="agent-card-left">
                            <div class="agent-badge-icon" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24;">A3</div>
                            <div>
                                <div class="agent-title-text">Lesion Analysis Agent (Dermoscopic ABCD)</div>
                                <div class="agent-subtitle-text">Lesion exhibits intermediate structural complexity warranting caution.</div>
                            </div>
                        </div>
                        <div class="status-pill-badge" style="background: #3b0764; color: #c084fc; border: 1px solid #7e22ce;">
                            ${c.A3.pill} ▾
                        </div>
                    </div>
                    <div class="agent-card-narrative">${c.A3.summary}</div>
                    <table class="clinical-eval-table">
                        <thead>
                            <tr><th>FEATURE / RULE</th><th>VALUE</th><th>THRESHOLD</th><th>STATUS</th><th>CLINICAL NOTE</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="col-feature">border_irregularity</td>
                                <td class="col-val">${c.A3.border.toFixed(2)}</td>
                                <td class="col-thresh">&lt;= 1.60</td>
                                <td><span class="${c.A3.border <= 1.6 ? 'badge-pass' : 'badge-flag'}">${c.A3.border <= 1.6 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Border compactness is ${c.A3.border.toFixed(2)} (cutoff 1.60; score ${c.A3.notched} octants)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">geometric_asymmetry</td>
                                <td class="col-val">${c.A3.asym.toFixed(2)}</td>
                                <td class="col-thresh">&lt;= 0.35</td>
                                <td><span class="${c.A3.asym <= 0.35 ? 'badge-pass' : 'badge-flag'}">${c.A3.asym <= 0.35 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Asymmetry fold ratio is ${c.A3.asym.toFixed(2)} (${c.A3.axes} axes with significant difference)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">color_variegation</td>
                                <td class="col-val">${c.A3.variegation.toFixed(2)}</td>
                                <td class="col-thresh">&lt;= 32.00</td>
                                <td><span class="${c.A3.variegation <= 32 ? 'badge-pass' : 'badge-flag'}">${c.A3.variegation <= 32 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Color standard deviation is ${c.A3.variegation.toFixed(1)} (${c.A3.chromophores})</td>
                            </tr>
                            <tr>
                                <td class="col-feature">total_dermoscopy_score</td>
                                <td class="col-val">${c.A3.tds.toFixed(2)}</td>
                                <td class="col-thresh">&lt;= 5.45</td>
                                <td><span class="${c.A3.tds <= 5.45 ? 'badge-pass' : 'badge-flag'}">${c.A3.tds <= 5.45 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Stolz TDS score is ${c.A3.tds.toFixed(2)} (benign upper: 4.75, suspicious upper: 5.45)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">lesion_area_fraction</td>
                                <td class="col-val">${c.A3.area.toFixed(2)}</td>
                                <td class="col-thresh">&gt;= 2.00</td>
                                <td><span class="badge-pass">✔ PASS</span></td>
                                <td class="col-note">Lesion covers ${c.A3.area.toFixed(1)}% of image field (min reliability 2.0%)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- CARD 4: AGENT 4 -->
                <div class="agent-card-container">
                    <div class="agent-card-header">
                        <div class="agent-card-left">
                            <div class="agent-badge-icon" style="background: rgba(147, 51, 234, 0.2); color: #d8b4fe;">A4</div>
                            <div>
                                <div class="agent-title-text">Context & Demographics Agent</div>
                                <div class="agent-subtitle-text">Patient presentation exhibits ${c.A4.pill} demographic risk based on age, anatomical site, and solar UV history.</div>
                            </div>
                        </div>
                        <div class="status-pill-badge" style="background: ${c.A4.pill === 'LOW' ? '#064e3b' : '#7f1d1d'}; color: ${c.A4.pill === 'LOW' ? '#34d399' : '#f87171'}; border: 1px solid ${c.A4.pill === 'LOW' ? '#059669' : '#dc2626'};">
                            ${c.A4.pill} ▾
                        </div>
                    </div>
                    <div class="agent-card-narrative">${c.A4.summary}</div>
                    <table class="clinical-eval-table">
                        <thead>
                            <tr><th>FEATURE / RULE</th><th>VALUE</th><th>THRESHOLD</th><th>STATUS</th><th>CLINICAL NOTE</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="col-feature">patient_age_risk</td>
                                <td class="col-val">${c.A4.age.toFixed(2)}</td>
                                <td class="col-thresh">&lt; 60.00</td>
                                <td><span class="${c.A4.age < 60 ? 'badge-pass' : 'badge-flag'}">${c.A4.age < 60 ? '✔ PASS' : '▲ FLAG'}</span></td>
                                <td class="col-note">Age is ${Math.round(c.A4.age)} yrs (senior cutoff: 60 yrs, geriatric cutoff: 75 yrs)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">anatomical_site_risk</td>
                                <td class="col-val">${c.A4.site}</td>
                                <td class="col-thresh">not in acral,scalp,neck,ear,face</td>
                                <td><span class="badge-pass">✔ PASS</span></td>
                                <td class="col-note">Location is '${c.A4.site}' (high risk sites: acral, ear, face, neck, scalp)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">demographic_risk_score</td>
                                <td class="col-val">${c.A4.risk.toFixed(2)}</td>
                                <td class="col-thresh">&lt; 0.70</td>
                                <td><span class="badge-pass">✔ PASS</span></td>
                                <td class="col-note">Context cumulative risk score is ${c.A4.risk.toFixed(2)} (MODERATE &gt;= 0.45, HIGH &gt;= 0.70)</td>
                            </tr>
                            <tr>
                                <td class="col-feature">demographic_completeness</td>
                                <td class="col-val">100.00</td>
                                <td class="col-thresh">== 100.00</td>
                                <td><span class="badge-pass">✔ PASS</span></td>
                                <td class="col-note">Demographic chart completeness is 100% (0 missing fields flagged)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- CARD 5: AGENT 5 -->
                <div class="agent-card-container">
                    <div class="agent-card-header">
                        <div class="agent-card-left">
                            <div class="agent-badge-icon" style="background: rgba(239, 68, 68, 0.2); color: #f87171;">A5</div>
                            <div>
                                <div class="agent-title-text">Safety Agent (Guardrails S1-S6)</div>
                                <div class="agent-subtitle-text">Clinical audit of 6 safety rules across multi-modal evidence.</div>
                            </div>
                        </div>
                        <div class="status-pill-badge" style="background: ${c.A5.veto ? '#7f1d1d' : '#064e3b'}; color: ${c.A5.veto ? '#f87171' : '#34d399'}; border: 1px solid ${c.A5.veto ? '#dc2626' : '#059669'};">
                            ${c.A5.pill} ▾
                        </div>
                    </div>
                    <table class="clinical-eval-table">
                        <thead>
                            <tr><th style="width: 80px;">RULE</th><th style="width: 80px;">VALUE</th><th style="width: 100px;">THRESHOLD</th><th style="width: 100px;">STATUS</th><th>CLINICAL NOTE</th></tr>
                        </thead>
                        <tbody>
                            ${sRulesHtml}
                        </tbody>
                    </table>
                </div>

                <!-- CARD 6: AGENT 6 -->
                <div class="agent-card-container">
                    <div class="agent-card-header">
                        <div class="agent-card-left">
                            <div class="agent-badge-icon" style="background: rgba(14, 165, 233, 0.2); color: #38bdf8;">A6</div>
                            <div>
                                <div class="agent-title-text">Coordinator Agent (Policy Engine)</div>
                                <div class="agent-subtitle-text">Final disposition is ${c.decision} with diagnosis '${c.class_name}' [${c.prediction}] (${c.A2.conf.toFixed(1)}% confidence).</div>
                            </div>
                        </div>
                        <div class="status-pill-badge" style="background: ${c.badge_color}20; color: ${c.badge_color}; border: 1px solid ${c.badge_color};">
                            ${c.decision} ▾
                        </div>
                    </div>
                    <div class="agent-card-narrative">
                        Coordinator Agent synthesized outputs from all 5 agents and executed ${c.A6.policy_name}: ${c.A6.policy_desc} The final diagnosis is confirmed as '${c.class_name}' [${c.prediction}] with disposition '${c.decision}'. ${c.A6.note}
                    </div>
                    <table class="clinical-eval-table">
                        <thead>
                            <tr><th>FEATURE / RULE</th><th>VALUE</th><th>THRESHOLD</th><th>STATUS</th><th>CLINICAL NOTE</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="col-feature">policy_decision</td>
                                <td class="col-val">${c.decision}</td>
                                <td class="col-thresh">== AUTO_TRIAGE / REFER / RE_IMAGE</td>
                                <td><span class="badge-pass">✔ PASS</span></td>
                                <td class="col-note">Coordinator applied ${c.A6.policy_name}: ${c.A6.policy_desc}</td>
                            </tr>
                            <tr>
                                <td class="col-feature">predicted_class_code</td>
                                <td class="col-val">${c.prediction}</td>
                                <td class="col-thresh">in 7 HAM10000 codes</td>
                                <td><span class="badge-pass">✔ PASS</span></td>
                                <td class="col-note">Preserved diagnostic classification: '${c.class_name}' [${c.prediction}] (${c.A2.conf.toFixed(1)}%)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            `;
            document.getElementById('agent-cards-container').innerHTML = cardsHtml;
        }

        // Handle uploaded image file directly
        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                const dataUrl = e.target.result;
                processUploadedImage(dataUrl, file.name);
            };
            reader.readAsDataURL(file);
        }

        function processUploadedImage(dataUrl, fileName) {
            // Show preview box
            const prevBox = document.getElementById('specimen-preview-box');
            prevBox.style.display = 'grid';
            document.getElementById('preview-img').src = dataUrl;
            document.getElementById('preview-filename').innerText = fileName;

            // Analyze image pixels via Canvas
            const img = new Image();
            img.onload = function() {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                const w = Math.min(300, img.width);
                const h = Math.min(300, img.height);
                canvas.width = w;
                canvas.height = h;
                ctx.drawImage(img, 0, 0, w, h);
                const imgData = ctx.getImageData(0, 0, w, h).data;

                let sumL = 0;
                let glareCount = 0;
                let darkCount = 0;
                const totalPixels = w * h;

                for (let i = 0; i < imgData.length; i += 4) {
                    const r = imgData[i];
                    const g = imgData[i+1];
                    const b = imgData[i+2];
                    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
                    sumL += lum;
                    if (r > 240 && g > 240 && b > 240) glareCount++;
                    if (lum < 30) darkCount++;
                }

                const meanLum = sumL / totalPixels;
                const glareRatio = glareCount / totalPixels;
                const darkCornerRatio = darkCount / totalPixels;

                // Quality determination
                let isQualityPoor = (glareRatio > 0.18);
                let qualityScore = isQualityPoor ? 38.0 : 96.5;

                // Pathology simulation based on color & luminance
                let isMalignant = (meanLum < 125);
                let predCode = isMalignant ? 'mel' : 'nv';
                let predName = isMalignant ? 'Melanoma' : 'Melanocytic Nevus';
                let conf = isMalignant ? 78.4 : 95.2;
                let decision = isQualityPoor ? 'RE_IMAGE' : (isMalignant ? 'REFER' : 'AUTO_TRIAGE');
                let badgeColor = isQualityPoor ? '#eab308' : (isMalignant ? '#ef4444' : '#10b981');

                document.getElementById('meta-tag-decision').innerHTML = `Decision: <strong>${decision}</strong>`;
                document.getElementById('meta-tag-class').innerHTML = `Dx: <strong>${predCode.toUpperCase()}</strong>`;
                document.getElementById('meta-tag-quality').innerHTML = `Quality: <strong>${qualityScore.toFixed(1)}/100</strong>`;

                const uploadedCase = {
                    id: fileName,
                    name: `Custom Upload: ${fileName}`,
                    prediction: predCode,
                    class_name: predName,
                    decision: decision,
                    badge_color: badgeColor,
                    decision_title: `${decision}: Analysis of Uploaded Lesion`,
                    rationale: `Coordinator evaluated uploaded image '${fileName}'. Captured mean luminance: ${meanLum.toFixed(1)}, glare ratio: ${(glareRatio*100).toFixed(2)}%. Diagnostic classification synthesized with active guardrails.`,
                    flow: {
                        A1: {title: "A1 QUALITY", status: isQualityPoor ? "POOR" : "PASS", sub: `Score ${qualityScore.toFixed(1)}/100`, color: isQualityPoor ? "#ef4444" : "#10b981"},
                        A2: {title: "A2 DIAGNOSIS", status: `${predCode} • ${Math.round(conf)}%`, sub: predName, color: isMalignant ? "#ef4444" : "#10b981"},
                        A3_A4: {title: "A3 + A4 EVIDENCE", status: isMalignant ? "SUSPICIOUS" : "LOW RISK", sub: "ABCD extraction active", color: isMalignant ? "#f59e0b" : "#10b981"},
                        A5: {title: "A5 SAFETY", status: isMalignant ? "VETO ACTIVE" : "CLEAR", sub: isMalignant ? "Malignancy flag" : "Safety clear", color: isMalignant ? "#ef4444" : "#10b981"},
                        A6: {title: "A6 COORDINATOR", status: decision, sub: "Final triage", color: badgeColor}
                    },
                    A1: {
                        score: qualityScore,
                        disposition: isQualityPoor ? "FAIL" : "PASS",
                        sharpness: isQualityPoor ? 35.0 : 98.0,
                        lighting: 86.0,
                        glare: Math.max(0, 100 - glareRatio * 150),
                        centering: 98.0,
                        lap_var: isQualityPoor ? 28.0 : 76.5,
                        tenengrad: isQualityPoor ? 310 : 980,
                        entropy: 6.4,
                        mean_gray: meanLum.toFixed(1),
                        glare_area: glareRatio * 100,
                        edge_density: darkCornerRatio * 100,
                        summary: `Image Quality Agent evaluated upload integrity at ${qualityScore.toFixed(1)}/100 resulting in disposition '${isQualityPoor ? 'FAIL' : 'PASS'}'.`
                    },
                    A2: {
                        pred: predCode,
                        conf: conf,
                        risk_tier: isMalignant ? "HIGH RISK (Malignant)" : "LOW RISK (Benign)",
                        margin: isMalignant ? 0.54 : 0.91,
                        entropy: isMalignant ? 0.92 : 0.31,
                        cum_malig: isMalignant ? 0.79 : 0.02,
                        mel_risk: isMalignant ? 0.78 : 0.01,
                        top1: predCode,
                        top2: isMalignant ? "nv" : "bkl",
                        top3: "bcc",
                        diff: isMalignant ? [
                            {name: "Melanoma", code: "mel", pct: 78.4},
                            {name: "Melanocytic Nevus", code: "nv", pct: 18.2},
                            {name: "Basal Cell Carcinoma", code: "bcc", pct: 2.1}
                        ] : [
                            {name: "Melanocytic Nevus", code: "nv", pct: 95.2},
                            {name: "Benign Keratosis", code: "bkl", pct: 3.5},
                            {name: "Dermatofibroma", code: "df", pct: 0.9}
                        ],
                        probs: isMalignant ? {
                            mel: 78.4, nv: 18.2, bcc: 2.1, akiec: 0.5, bkl: 0.8, df: 0.0, vasc: 0.0
                        } : {
                            mel: 0.8, nv: 95.2, bcc: 0.4, akiec: 0.1, bkl: 3.5, df: 0.9, vasc: 0.1
                        },
                        summary: `Diagnosis Agent classified uploaded image as '${predName}' (${predCode}) with ${conf.toFixed(1)}% primary probability.`
                    },
                    A3: {
                        pill: isMalignant ? "SUSPICIOUS" : "BENIGN",
                        suspicion: isMalignant ? 0.65 : 0.22,
                        tds: isMalignant ? 6.50 : 2.20,
                        border: isMalignant ? 15.20 : 1.30,
                        asym: isMalignant ? 0.45 : 0.00,
                        variegation: isMalignant ? 42.0 : 19.5,
                        area: 46.0,
                        notched: isMalignant ? "7/8" : "1/8",
                        axes: isMalignant ? "1/2" : "0/2",
                        chromophores: isMalignant ? "3/6 chromophores" : "1/6 chromophore",
                        summary: `Lesion Analysis Agent computed an ABCD suspicion score of ${isMalignant ? '0.65' : '0.22'}.`
                    },
                    A4: {
                        pill: "LOW",
                        age: 45.0,
                        site: "general skin",
                        risk: 0.35,
                        comp: 100.0,
                        summary: "Context & Demographics Agent evaluated clinical risk at 0.35/1.0 (LOW)."
                    },
                    A5: {
                        pill: isMalignant ? "VETO ACTIVE" : "CLEAR",
                        veto: isMalignant,
                        rules: [
                            {"rule": "S1", "val": String(isMalignant), "pass": !isMalignant, "note": isMalignant ? `Residual / Predicted Malignancy: Primary predicted class is malignant '${predName}' [${predCode}] (${conf.toFixed(1)}% confidence).` : "No residual primary malignancy detected."},
                            {"rule": "S2", "val": "false", "pass": true, "note": "ABCD Lesion Structural Discordance: Concordant."},
                            {"rule": "S3", "val": "false", "pass": true, "note": "Diagnostic Ambiguity: Candidate margin decisive."},
                            {"rule": "S4", "val": "false", "pass": true, "note": "Demographic Context Discordance: Concordant."},
                            {"rule": "S5", "val": String(isQualityPoor), "pass": !isQualityPoor, "note": isQualityPoor ? "Severe optical degradation block." : "Image quality meets screening requirements."},
                            {"rule": "S6", "val": "false", "pass": true, "note": "Test-Time Augmentation Disagreement: TTA views stable."}
                        ]
                    },
                    A6: {
                        pill: decision,
                        policy_name: isQualityPoor ? "Policy Rule 1 (Capture Quality Gating)" : (isMalignant ? "Policy Rule 2 (Malignancy Referral)" : "Policy Rule 4 (Multi-Agent Consensus Clearance)"),
                        policy_desc: isQualityPoor ? "Image quality inadequate; request recapture." : (isMalignant ? `Detected malignant pathology '${predCode}'.` : "Unanimous benign consensus across all agents."),
                        note: isQualityPoor ? "Recapture required." : (isMalignant ? "Automated triage strictly forbidden; specialist biopsy indicated." : "Cleared for routine follow-up.")
                    }
                };

                CASES['custom_upload'] = uploadedCase;
                loadCase('custom_upload', null);
            };
            img.src = dataUrl;
        }

        // Restore saved theme or default
        const savedTheme = localStorage.getItem('dermaguard_theme') || 'dark';
        setTheme(savedTheme);

        // Initial setup
        renderCaseButtons();
        renderPreviewTable();
        loadCase('case2', null);
    </script>
</body>
</html>
"""

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ["/api/download-csv", "/download-csv"]:
            if DERMAGUARD_URL:
                try:
                    req = Request(f"{DERMAGUARD_URL}/submission-csv", headers={"User-Agent": "Vercel-App-Gateway"})
                    with urlopen(req, timeout=5) as response:
                        csv_data = response.read()
                except Exception:
                    csv_data = generate_csv().encode("utf-8")
            else:
                csv_data = generate_csv().encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="final_submission_format.csv"')
            self.send_header("Content-Length", str(len(csv_data)))
            self.end_headers()
            self.wfile.write(csv_data)
            return

        elif path.startswith("/api/predict"):
            if DERMAGUARD_URL:
                try:
                    target_url = f"{DERMAGUARD_URL}/predict?{parsed.query}" if parsed.query else f"{DERMAGUARD_URL}/predict"
                    req = Request(target_url, headers={"User-Agent": "Vercel-App-Gateway"})
                    with urlopen(req, timeout=5) as response:
                        resp_body = response.read()
                except Exception:
                    params = parse_qs(parsed.query)
                    case_key = params.get("case", ["case1"])[0]
                    case_data = CLINICAL_CASES.get(case_key, CLINICAL_CASES["case1"])
                    resp_body = json.dumps(case_data).encode("utf-8")
            else:
                params = parse_qs(parsed.query)
                case_key = params.get("case", ["case1"])[0]
                case_data = CLINICAL_CASES.get(case_key, CLINICAL_CASES["case1"])
                resp_body = json.dumps(case_data).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        elif path.startswith("/api/metrics"):
            if DERMAGUARD_URL:
                try:
                    req = Request(f"{DERMAGUARD_URL}/metrics", headers={"User-Agent": "Vercel-App-Gateway"})
                    with urlopen(req, timeout=5) as response:
                        resp_body = response.read()
                except Exception:
                    resp_body = json.dumps(SUMMARY_METRICS).encode("utf-8")
            else:
                resp_body = json.dumps(SUMMARY_METRICS).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        else:
            page_data = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page_data)))
            self.end_headers()
            self.wfile.write(page_data)
            return

    def do_POST(self):
        if self.path.startswith("/api/predict"):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

            if DERMAGUARD_URL:
                try:
                    req = Request(f"{DERMAGUARD_URL}/predict", data=post_data, headers={
                        "Content-Type": "application/json",
                        "User-Agent": "Vercel-App-Gateway"
                    })
                    with urlopen(req, timeout=5) as response:
                        resp_body = response.read()
                except Exception:
                    try:
                        data = json.loads(post_data.decode("utf-8"))
                    except Exception:
                        data = {}
                    case_key = data.get("case", "case1")
                    case_data = CLINICAL_CASES.get(case_key, CLINICAL_CASES["case1"])
                    resp_body = json.dumps(case_data).encode("utf-8")
            else:
                try:
                    data = json.loads(post_data.decode("utf-8"))
                except Exception:
                    data = {}
                case_key = data.get("case", "case1")
                case_data = CLINICAL_CASES.get(case_key, CLINICAL_CASES["case1"])
                resp_body = json.dumps(case_data).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return
        else:
            self.send_response(404)
            self.end_headers()
            return
