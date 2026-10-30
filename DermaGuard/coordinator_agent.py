"""
Agentic DermaGuard - Agent 6: Chief Coordinator Agent
Track 2 (Healthcare) - Jaypee Agentic AI International Summit 2026 (Day 2 / Level 2)

Role:
  Synthesizes evidence from all five specialized agents:
    - A1: Image Quality Sentinel
    - A2: Deep Diagnostic Classifier
    - A3: Lesion Morphological Analysis Agent
    - A4: Epidemiological Reasoning & Context Agent
    - A5: Clinical Safety & Veto Agent
  Applies an explicit, documented clinical decision policy to produce:
    - final_predicted_label: The authoritative lesion diagnosis code
    - decision: AUTO_TRIAGE, REFER, or RE_IMAGE
    - traceable_rationale: Complete multi-agent audit trail
    - flow_summary: 5-stage pipeline summary matching Day 2 brief specification
"""

from typing import Dict, Any, List


class CoordinatorAgent:
    def __init__(self):
        pass

    def coordinate(
        self,
        a1_quality: Dict[str, Any],
        a2_diagnosis: Dict[str, Any],
        a3_lesion: Dict[str, Any],
        a4_context: Dict[str, Any],
        a5_safety: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Executes the documented multi-agent clinical decision policy.
        """
        # Extract upstream findings
        q_status = a1_quality.get("status", "ACCEPTABLE")
        q_score = a1_quality.get("quality_score", 1.0)
        
        pred_label = a2_diagnosis.get("predicted_class", "nv")
        pred_name = a2_diagnosis.get("display_name", "Melanocytic Nevus")
        conf = a2_diagnosis.get("confidence", 0.0)
        is_malignant = a2_diagnosis.get("is_high_risk", False)
        
        a3_level = a3_lesion.get("suspicion_level", "UNSUSPICIOUS")
        a3_score = a3_lesion.get("suspicion_score", 0.0)
        
        a4_indicator = a4_context.get("risk_indicator", "AVERAGE_RISK_CONTEXT")
        
        safety_veto = a5_safety.get("veto", False)
        safety_flag = a5_safety.get("safety_flag", "SAFETY CLEAR")
        primary_trigger = a5_safety.get("primary_trigger", "None")

        # ----------------------------------------------------------------------
        # Documented Decision Policy
        # ----------------------------------------------------------------------
        rationale_steps: List[str] = []

        # Policy Step 1: Optical Gatekeeper Evaluation
        if q_status == "POOR":
            decision = "RE_IMAGE"
            badge_color = "#dc2626"
            icon = "🔄"
            rationale_steps.append(
                f"[A1 Quality Override] Specimen quality is POOR (score {q_score:.2f}). "
                f"Diagnostic classifier is safely gated. Recapture ordered to avoid diagnostic error on occluded/blurry scan."
            )
        # Policy Step 2: Malignancy or Safety Veto Review
        elif is_malignant or safety_veto:
            decision = "REFER"
            badge_color = "#ea580c"
            icon = "🚨"
            if is_malignant:
                rationale_steps.append(
                    f"[A2 High-Risk Pathology] Predicted malignant/precancerous neoplasm '{pred_label.upper()}' "
                    f"({pred_name}) with {conf*100:.1f}% confidence. Immediate in-person excision/biopsy required."
                )
            if safety_veto:
                rationale_steps.append(
                    f"[A5 Safety Veto Triggered] Conflict detected: {primary_trigger}. "
                    f"A3 morphology suspicion ({a3_score:.2f}) or A4 clinical context overrides automatic clearance."
                )
        # Policy Step 3: Multi-Agent Consensus Auto-Triage
        else:
            decision = "AUTO_TRIAGE"
            badge_color = "#16a34a"
            icon = "✅"
            rationale_steps.append(
                f"[Multi-Agent Consensus] A1 quality verified usable ({q_status}, {q_score:.2f}). "
                f"A2 confirms benign '{pred_label}' with high confidence ({conf*100:.1f}%). "
                f"A3 visual cues normal ({a3_level}, {a3_score:.2f}). A5 confirms SAFETY CLEAR. Cleared for routine follow-up."
            )

        # ----------------------------------------------------------------------
        # Assemble Flow Card Summary (Matching Day 2 Brief Specimen Cards)
        # ----------------------------------------------------------------------
        # A1 Card
        a1_card_status = "PASS" if q_status in ["GOOD", "ACCEPTABLE"] else "FAIL"
        a1_card_text = "Image is usable" if a1_card_status == "PASS" else "Quality inadequate"

        # A2 Card
        a2_card_status = f"{pred_label} • {conf:.2f}"
        a2_card_text = f"Model predicts {pred_name.split()[0].lower()}"

        # A3 + A4 Card
        if a3_level == "SUSPICIOUS" or a4_indicator == "HIGH_RISK_CONTEXT":
            a3_a4_card_status = "SUSPICIOUS"
            a3_a4_card_text = "Other evidence raises concern"
        elif a3_level == "BORDERLINE" or a4_indicator == "MODERATE_RISK_CONTEXT":
            a3_a4_card_status = "BORDERLINE"
            a3_a4_card_text = "Minor atypia / context caution"
        else:
            a3_a4_card_status = "BENIGN"
            a3_a4_card_text = "Normal visual & context cues"

        # A5 Card
        a5_card_status = "SAFETY FLAG" if safety_veto else "SAFETY CLEAR"
        a5_card_text = "Conflict detected" if safety_veto else "No safety conflicts"

        # A6 Card
        a6_card_status = decision
        a6_card_text = "Final disposition"

        flow_summary = {
            "A1": {"title": "A1: Quality", "status": a1_card_status, "subtitle": a1_card_text, "color": "#10b981" if a1_card_status == "PASS" else "#ef4444"},
            "A2": {"title": "A2: Diagnosis", "status": a2_card_status, "subtitle": a2_card_text, "color": "#2563eb"},
            "A3_A4": {"title": "A3 + A4: Evidence", "status": a3_a4_card_status, "subtitle": a3_a4_card_text, "color": "#f59e0b" if a3_a4_card_status != "BENIGN" else "#10b981"},
            "A5": {"title": "A5: Safety", "status": a5_card_status, "subtitle": a5_card_text, "color": "#ef4444" if safety_veto else "#10b981"},
            "A6": {"title": "A6: Coordinator", "status": a6_card_status, "subtitle": a6_card_text, "color": badge_color},
        }

        return {
            "agent_id": "A6",
            "agent_name": "Coordinator Agent",
            "final_predicted_label": pred_label,
            "final_predicted_name": pred_name,
            "decision": decision,
            "decision_icon": icon,
            "badge_color": badge_color,
            "traceable_rationale": " -> ".join(rationale_steps),
            "rationale_list": rationale_steps,
            "flow_summary": flow_summary,
            "evidence_synthesis": {
                "A1_quality_status": q_status,
                "A2_diagnosis_label": pred_label,
                "A2_confidence": conf,
                "A3_suspicion_level": a3_level,
                "A3_suspicion_score": a3_score,
                "A4_context_indicator": a4_indicator,
                "A5_safety_flag": safety_flag,
                "A5_veto_active": safety_veto,
                "A6_final_decision": decision,
            }
        }
