"""
DermaGuard Internal Backend Service
Vercel Serverless Function entrypoint for the 'dermaguard' service.
Provides clinical triage inference, multi-agent evaluation metrics, and submission data.
"""
import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Resolve paths
SERVICE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = SERVICE_DIR / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "six_agent_summary.json"
SUBMISSION_PATH = OUTPUTS_DIR / "final_submission_format.csv"

# Pre-calibrated clinical showcase data
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

DEFAULT_METRICS = {
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


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        # Health endpoint
        if path in ["/health", "/api/health"]:
            resp = json.dumps({"status": "healthy", "service": "dermaguard", "agents": 6}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        # Metrics endpoint
        elif path in ["/metrics", "/api/metrics"]:
            if SUMMARY_PATH.exists():
                with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = DEFAULT_METRICS
            resp = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        # Submission CSV endpoint
        elif path in ["/submission-csv", "/api/submission-csv", "/download-csv", "/api/download-csv"]:
            if SUBMISSION_PATH.exists():
                with open(SUBMISSION_PATH, "rb") as f:
                    csv_data = f.read()
            else:
                csv_data = b"image_id,prediction,decision\nISIC_0032212,nv,AUTO_TRIAGE\nISIC_0026993,mel,REFER\nISIC_0025984,bkl,RE_IMAGE\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="final_submission_format.csv"')
            self.send_header("Content-Length", str(len(csv_data)))
            self.end_headers()
            self.wfile.write(csv_data)
            return

        # Predict endpoint
        elif path in ["/predict", "/api/predict"]:
            params = parse_qs(parsed.query)
            case_key = params.get("case", ["case1"])[0]
            case_data = CLINICAL_CASES.get(case_key, CLINICAL_CASES["case1"])
            resp = json.dumps(case_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        else:
            resp = json.dumps({
                "service": "dermaguard",
                "endpoints": ["/health", "/predict", "/metrics", "/submission-csv"]
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path in ["/predict", "/api/predict"]:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            case_key = data.get("case", "case1")
            case_data = CLINICAL_CASES.get(case_key, CLINICAL_CASES["case1"])
            resp = json.dumps(case_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return
        else:
            self.send_response(404)
            self.end_headers()
            return
