"""Inference: loads the trained artifacts and scores messages."""

from __future__ import annotations

import json
import re
import threading
from functools import lru_cache

import joblib
import numpy as np

from ..config import Settings, get_settings
from ..core.features import analyse_message, rule_score
from ..core.links import analyse_links_in
from .calibrator import meta_predict

#: Verdict wording. The numeric cut-points are derived from the *learned*
#: threshold rather than hardcoded, so a band can never disagree with the
#: is_scam decision.
BAND_ADVICE = {
    "dangerous": "Do not reply. Block and report this contact.",
    "likely_scam": "Treat as a scam. Do not click links or send money.",
    "suspicious": "Be careful. Verify the sender through a channel you trust.",
    "safe": "No scam indicators found.",
}

CATEGORY_LABELS = {
    "phishing_bank_kyc": "Bank / KYC phishing",
    "lottery_prize": "Lottery or prize scam",
    "job_task_scam": "Fake job or task scam",
    "investment_crypto": "Investment / crypto fraud",
    "otp_credential_theft": "OTP or credential theft",
    "tech_support": "Fake tech support",
    "romance_pig_butchering": "Romance / long-con fraud",
    "delivery_customs": "Parcel or customs scam",
    "family_impersonation": "Family impersonation",
    "loan_offer": "Fake loan offer",
    "refund_overpayment": "Refund / overpayment scam",
    "charity_fraud": "Fake charity appeal",
    "digital_arrest": "Digital arrest / fake police",
    "business_impersonation": "Fake business / brand impersonation",
    "course_mentorship_scam": "Fake course / mentorship upsell",
    "extortion_ransom": "Extortion / ransom demand",
    "trusted_contact_link": "Malicious link from a 'trusted' contact",
}


class ModelNotTrained(RuntimeError):
    """Raised when no trained artifact is available."""


def build_bands(threshold: float, strict: float) -> list[tuple[float, str, str]]:
    strict = max(strict, threshold + 0.05)
    return [
        (strict, "dangerous", BAND_ADVICE["dangerous"]),
        (threshold, "likely_scam", BAND_ADVICE["likely_scam"]),
        (threshold * 0.55, "suspicious", BAND_ADVICE["suspicious"]),
        (0.0, "safe", BAND_ADVICE["safe"]),
    ]


class ScamDetector:
    """Thread-safe wrapper around the persisted pipeline."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        path = self.settings.model_path
        if not path.exists():
            raise ModelNotTrained(f"No model at {path}. Run:  scamshield-train")
        bundle = joblib.load(path)
        self._lock = threading.Lock()
        self.members = bundle["members"]
        self.kind = bundle["kind"]
        self.category_model = bundle["category_model"]
        self.alpha = float(bundle["alpha"])
        # Older artifacts predate the meta-calibrator; fall back to the fixed
        # blend rather than refusing to load them.
        self.meta_model = bundle.get("meta_model")
        self.fusion = bundle.get("fusion", "blend") if self.meta_model is not None else "blend"
        self.threshold = float(self.settings.threshold_override or bundle["threshold"])
        self.strict_threshold = float(bundle["strict_threshold"])
        self.bands = build_bands(self.threshold, self.strict_threshold)
        meta = self.settings.metadata_path
        self.metadata = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}

    # -- scoring -----------------------------------------------------------
    def _ml_proba(self, texts: list[str]) -> np.ndarray:
        return np.mean([m.predict_proba(texts)[:, 1] for m in self.members], axis=0)

    def score_batch(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        cfg = self.settings
        with self._lock:
            p_ml = self._ml_proba(texts)

        rule_results = [rule_score(t) for t in texts]
        rule_scores = np.array([r[0] for r in rule_results])
        if self.fusion == "meta":
            with self._lock:
                fused = meta_predict(self.meta_model, texts, p_ml, rule_scores)
        else:
            fused = np.clip((1 - self.alpha) * p_ml + self.alpha * rule_scores, 0, 1)

        results = []
        for text, ml, (rules, signals), score in zip(texts, p_ml, rule_results, fused, strict=True):
            final = float(score)

            # "Hi" is not evidence of anything. Without this guard the model
            # extrapolates from openers common in scam templates and flags
            # ordinary greetings.
            low_info = len(re.findall(r"\w+", text)) < cfg.low_info_tokens and not signals
            if low_info:
                final = min(final, cfg.low_info_cap)

            verdict, advice = next((v, a) for t, v, a in self.bands if final >= t)

            category = category_label = None
            confidence = 0.0
            if final >= self.threshold or rules >= 0.5:
                with self._lock:
                    probs = self.category_model.predict_proba([text])[0]
                    classes = self.category_model.classes_
                idx = int(np.argmax(probs))
                # Playbooks overlap heavily; asserting one from a near-uniform
                # posterior would be misleading.
                if float(probs[idx]) >= cfg.min_category_confidence:
                    category = str(classes[idx])
                    category_label = CATEGORY_LABELS.get(category, category.replace("_", " ").title())
                    confidence = round(float(probs[idx]), 4)

            results.append({
                "text": text,
                "risk_score": round(final, 4),
                "ml_score": round(float(ml), 4),
                "rule_score": round(float(rules), 4),
                "is_scam": bool(final >= self.threshold),
                "high_confidence": bool(final >= self.strict_threshold),
                "low_information": bool(low_info),
                "verdict": verdict,
                "advice": advice,
                "category": category,
                "category_label": category_label,
                "category_confidence": confidence,
                "signals": [s.as_dict() for s in signals[:8]],
                # Pre-click: every link is analysed offline, so the user can
                # see where it goes without opening it.
                "links": [r.as_dict() for r in analyse_links_in(text)],
            })
        return results

    def score(self, text: str) -> dict:
        return self.score_batch([text])[0]

    def explain(self, text: str) -> dict:
        """Score plus the raw numeric feature vector, for debugging."""
        result = self.score(text)
        result["features"] = {k: round(v, 4) for k, v in analyse_message(text)[0].items() if v}
        return result

    def info(self) -> dict:
        md = self.metadata
        return {
            "model": self.kind,
            "threshold": self.threshold,
            "strict_threshold": self.strict_threshold,
            "blend_alpha": self.alpha,
            "fusion": self.fusion,
            "bands": [{"min": round(t, 4), "verdict": v} for t, v, _ in self.bands],
            "trained_at": md.get("trained_at"),
            "corpus_size": md.get("corpus_size"),
            "template_groups": md.get("template_groups"),
            "categories": md.get("categories", sorted(CATEGORY_LABELS)),
            "holdout_test": md.get("holdout_test", {}).get("blended"),
            "category_cv_accuracy": md.get("category_cv_accuracy"),
        }


@lru_cache(maxsize=1)
def get_detector() -> ScamDetector:
    return ScamDetector()


def reset_detector() -> None:
    """Drop the cached detector so a freshly trained model is picked up."""
    get_detector.cache_clear()
