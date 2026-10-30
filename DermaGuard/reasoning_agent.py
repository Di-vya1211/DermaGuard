"""
Agentic DermaGuard - Agent 4: Reasoning / Context Agent
Track 2 (Healthcare) - Jaypee Agentic AI International Summit 2026 (Day 2 / Level 2)

Role:
  Evaluates patient epidemiological context (age, gender/sex, anatomical localization)
  to modulate clinical screening risk without demographic bias.
  Robustly handles missing fields (imputation flags and uncertainty bounds).

Outputs:
  - context_evidence: Detailed clinical epidemiology rationale
  - risk_indicator: Categorical risk tier (HIGH_RISK_CONTEXT, MODERATE_RISK_CONTEXT, AVERAGE_RISK_CONTEXT)
  - missing_field_flags: Boolean flags documenting field completeness
"""

import math
from typing import Dict, Any, Optional, Union


# Anatomic site risk stratification based on dermato-oncology epidemiology
HIGH_RISK_SITES = {"face", "scalp", "neck", "ear", "foot", "hand", "acral", "oral / genital"}
MODERATE_RISK_SITES = {"back", "trunk", "chest", "upper extremity"}
LOW_RISK_SITES = {"abdomen", "lower extremity"}


class ReasoningContextAgent:
    def __init__(self):
        pass

    def evaluate_context(
        self,
        age: Optional[Union[float, int, str]] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates clinical epidemiological risk given age, gender, and anatomic localization.
        Gracefully handles missing, NaN, or corrupted values with explicit flags.
        """
        missing_flags = {
            "age_missing": False,
            "sex_missing": False,
            "localization_missing": False,
        }
        
        # 1. Parse & validate Age
        parsed_age = None
        if age is None or (isinstance(age, float) and math.isnan(age)):
            missing_flags["age_missing"] = True
        else:
            try:
                parsed_age = float(age)
                if parsed_age < 0 or parsed_age > 125 or math.isnan(parsed_age):
                    missing_flags["age_missing"] = True
                    parsed_age = None
            except (ValueError, TypeError):
                missing_flags["age_missing"] = True

        # 2. Parse & validate Sex / Gender
        parsed_sex = None
        if sex is None or (isinstance(sex, float) and math.isnan(sex)):
            missing_flags["sex_missing"] = True
        else:
            s_str = str(sex).strip().lower()
            if s_str in {"male", "m"}:
                parsed_sex = "male"
            elif s_str in {"female", "f"}:
                parsed_sex = "female"
            else:
                missing_flags["sex_missing"] = True

        # 3. Parse & validate Localization
        parsed_loc = None
        if localization is None or (isinstance(localization, float) and math.isnan(localization)):
            missing_flags["localization_missing"] = True
        else:
            l_str = str(localization).strip().lower()
            if l_str in {"unknown", "nan", ""}:
                missing_flags["localization_missing"] = True
            else:
                parsed_loc = l_str

        # 4. Compute Contextual Risk Factor
        # Baseline population risk score
        risk_score = 0.30
        evidence_points = []

        # Age Contribution
        if missing_flags["age_missing"]:
            evidence_points.append("Patient age unavailable; defaulted to baseline population epidemiological prior.")
        else:
            if parsed_age >= 65:
                risk_score += 0.30
                evidence_points.append(f"Advanced patient age ({int(parsed_age)}y) correlates with elevated melanoma & BCC incidence.")
            elif parsed_age >= 50:
                risk_score += 0.15
                evidence_points.append(f"Patient age ({int(parsed_age)}y) in elevated risk cohort (>50y).")
            elif parsed_age < 30:
                risk_score -= 0.10
                evidence_points.append(f"Young patient age ({int(parsed_age)}y); lower baseline malignancy prevalence.")
            else:
                evidence_points.append(f"Patient age: {int(parsed_age)}y (intermediate cohort).")

        # Anatomical Site Contribution
        if missing_flags["localization_missing"]:
            evidence_points.append("Lesion anatomical site unspecified.")
        else:
            if any(site in parsed_loc for site in HIGH_RISK_SITES):
                risk_score += 0.25
                evidence_points.append(f"High-risk anatomic site ('{parsed_loc}'): chronic ultraviolet photo-damage exposure area.")
            elif any(site in parsed_loc for site in MODERATE_RISK_SITES):
                risk_score += 0.10
                evidence_points.append(f"Intermediate anatomical localization ('{parsed_loc}').")
            else:
                evidence_points.append(f"Standard anatomical site ('{parsed_loc}').")

        # Sex Contribution
        if not missing_flags["sex_missing"]:
            if parsed_sex == "male" and parsed_age and parsed_age >= 55:
                risk_score += 0.05
                evidence_points.append("Male gender with age >=55y carries statistically higher melanoma mortality hazard.")

        risk_score = float(max(0.05, min(0.95, risk_score)))

        # 5. Determine Categorical Risk Indicator
        if risk_score >= 0.60:
            risk_indicator = "HIGH_RISK_CONTEXT"
        elif risk_score >= 0.40:
            risk_indicator = "MODERATE_RISK_CONTEXT"
        else:
            risk_indicator = "AVERAGE_RISK_CONTEXT"

        missing_count = sum(missing_flags.values())
        missing_flags["missing_count"] = missing_count
        missing_flags["complete_context"] = (missing_count == 0)

        return {
            "agent_id": "A4",
            "agent_name": "Reasoning / Context Agent",
            "risk_indicator": risk_indicator,
            "context_risk_score": round(risk_score, 3),
            "context_evidence": evidence_points,
            "patient_context": {
                "age": parsed_age,
                "sex": parsed_sex,
                "localization": parsed_loc,
            },
            "missing_field_flags": missing_flags,
        }
