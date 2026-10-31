"""
Agentic DermaGuard - Agent 5: Safety Agent
Track 2 (Healthcare) - Jaypee Agentic AI International Summit 2026

Role:
  Acts as an autonomous clinical safety auditor.
  Reviews outputs from Agent 1 (Quality), Agent 2 (Diagnosis), Agent 3 (Lesion Analysis),
  and Agent 4 (Context) to actively detect conflicts, borderline uncertainty, or unsafe automatic discharges.

Outputs:
  - safety_flag: Categorical flag ('SAFETY FLAG' or 'SAFETY CLEAR')
  - veto: Boolean flag (True = veto automatic triage; force referral or re-image)
  - trigger: Specific reason trigger that activated the safety flag
  - justification: Thorough medical safety rationale documenting clinical risk
"""

from typing import Dict, Any, List


HIGH_RISK_CLASSES = ["mel", "bcc", "akiec"]


class SafetyAgent:
    def __init__(self):
        pass

    def audit(
        self,
        a1_quality: Dict[str, Any],
        a2_diagnosis: Dict[str, Any],
        a3_lesion: Dict[str, Any],
        a4_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Audits all upstream evidence for clinical contradictions and safety risks.
        Returns safety flag, veto status, active triggers, and clinical justification.
        """
        triggers: List[str] = []
        justifications: List[str] = []

        q_status = a1_quality.get("status", "ACCEPTABLE")
        q_score = a1_quality.get("quality_score", 1.0)
        
        pred_class = a2_diagnosis.get("predicted_class", "nv")
        conf = a2_diagnosis.get("confidence", 0.0)
        probs = a2_diagnosis.get("probabilities", {})
        is_malignant = pred_class in HIGH_RISK_CLASSES

        suspicion_score = a3_lesion.get("suspicion_score", 0.0)
        asym_score = a3_lesion.get("asymmetry_score", 0.0)
        border_score = a3_lesion.get("border_irregularity_score", 0.0)

        context_risk = a4_context.get("risk_indicator", "AVERAGE_RISK_CONTEXT")
        patient_ctx = a4_context.get("patient_context", {})
        age = patient_ctx.get("age")
        loc = str(patient_ctx.get("localization") or "")

        # ----------------------------------------------------------------------
        # Check 1: Severe Image Quality Failure
        # ----------------------------------------------------------------------
        if q_status == "POOR":
            triggers.append("CRITICAL_QUALITY_FAILURE")
            justifications.append(
                f"Image Quality Sentinel scored POOR ({q_score:.2f}). Running autonomous screening on degraded "
                f"dermoscopy introduces unacceptable false-reassurance risk."
            )

        # ----------------------------------------------------------------------
        # Check 2: Direct Malignancy Identification
        # ----------------------------------------------------------------------
        if is_malignant:
            triggers.append("HIGH_RISK_PATHOLOGY_CONFIRMED")
            justifications.append(
                f"Agent 2 diagnosed high-risk malignant/precancerous neoplasm '{pred_class.upper()}' "
                f"({a2_diagnosis.get('display_name')}) with {conf*100:.1f}% confidence. Autonomous discharge is strictly prohibited."
            )

        # ----------------------------------------------------------------------
        # Check 3: Benign Prediction vs Suspicious Lesion Morphology (A2 vs A3 Conflict)
        # ----------------------------------------------------------------------
        if not is_malignant and (suspicion_score >= 0.52 or asym_score >= 0.65 or border_score >= 0.60):
            triggers.append("BENIGN_MORPHOLOGY_DISCREPANCY")
            justifications.append(
                f"Conflict detected between Agent 2 (predicts benign '{pred_class}') and Agent 3 "
                f"(morphological suspicion score {suspicion_score:.2f}, asymmetry {asym_score:.2f}, border {border_score:.2f}). "
                f"Atypical features warrant specialist dermoscopic review to rule out amelanotic or early dysplastic melanoma."
            )

        # ----------------------------------------------------------------------
        # Check 4: Benign Prediction in High-Risk Clinical Context (A2 vs A4 Conflict)
        # ----------------------------------------------------------------------
        if not is_malignant and context_risk == "HIGH_RISK_CONTEXT":
            if (age and age >= 65) or any(site in loc for site in ["face", "scalp", "ear", "neck"]):
                triggers.append("HIGH_RISK_EPIDEMIOLOGICAL_CONTEXT")
                justifications.append(
                    f"Patient demographic context is high-risk (Age: {age or 'N/A'}, Site: '{loc}'). "
                    f"Chronic photo-damaged anatomical zones carry high rates of lentigo maligna and amelanotic BCC."
                )

        # ----------------------------------------------------------------------
        # Check 5: Diagnostic Uncertainty / Borderline Classification
        # ----------------------------------------------------------------------
        # Sum of probabilities across all 3 malignant classes
        malignant_prob = float(probs.get("mel", 0.0) + probs.get("bcc", 0.0) + probs.get("akiec", 0.0))
        if conf < 0.65 or malignant_prob >= 0.18:
            triggers.append("DIAGNOSTIC_AMBIGUITY_ELEVATED_MALIGNANCY_TAIL")
            justifications.append(
                f"Diagnostic classifier exhibits uncertainty (confidence {conf*100:.1f}%, malignant tail risk {malignant_prob*100:.1f}%). "
                f"Decision threshold requires specialist human review."
            )

        # ----------------------------------------------------------------------
        # Final Safety Evaluation
        # ----------------------------------------------------------------------
        veto_active = len(triggers) > 0
        safety_flag = "SAFETY FLAG" if veto_active else "SAFETY CLEAR"

        if not veto_active:
            justifications.append(
                f"Multi-agent consensus confirmed: Image quality is {q_status}, diagnosis is benign '{pred_class}' "
                f"with high confidence ({conf*100:.1f}%), morphological suspicion is low ({suspicion_score:.2f}), "
                f"and context profile is standard. Cleared for safe automated triage."
            )

        return {
            "agent_id": "A5",
            "agent_name": "Safety Agent",
            "safety_flag": safety_flag,
            "veto": veto_active,
            "triggers": triggers,
            "primary_trigger": triggers[0] if triggers else "NONE (All Checks Cleared)",
            "justification": " | ".join(justifications),
            "justifications_list": justifications,
        }
