"""The learned meta-calibrator and its fallback."""

import numpy as np

from scamshield.ml.calibrator import (
    META_FEATURE_NAMES,
    build_meta_model,
    meta_features,
    meta_predict,
)
from scamshield.ml.detector import get_detector

TEXTS = [
    "Please share the OTP sent to your phone",
    "hey are we still on for lunch tomorrow?",
    "Your KYC expired, verify at http://sbi-kyc.xyz",
    "thanks, see you at 6",
]


def test_meta_features_shape_and_order():
    X = meta_features(TEXTS, np.array([0.9, 0.1, 0.95, 0.02]))
    assert X.shape == (len(TEXTS), len(META_FEATURE_NAMES))
    assert X[0][0] == 0.9 and X[1][0] == 0.1     # ml_score is column 0
    assert np.isfinite(X).all()


def test_meta_features_accept_precomputed_rule_scores():
    ml = np.array([0.9, 0.1, 0.95, 0.02])
    rules = np.array([0.8, 0.0, 0.7, 0.0])
    X = meta_features(TEXTS, ml, rules)
    assert list(X[:, 1]) == list(rules)


def test_meta_model_trains_and_predicts_in_range():
    ml = np.array([0.9, 0.1, 0.95, 0.02])
    y = np.array([1, 0, 1, 0])
    model = build_meta_model().fit(meta_features(TEXTS, ml), y)
    p = meta_predict(model, TEXTS, ml)
    assert p.shape == (4,)
    assert ((p >= 0) & (p <= 1)).all()


def test_detector_reports_its_fusion_mode():
    d = get_detector()
    assert d.fusion in {"meta", "blend"}
    assert d.info()["fusion"] == d.fusion


def test_detector_falls_back_when_no_meta_model():
    """Artifacts trained before the calibrator existed must still load."""
    d = get_detector()
    assert d.fusion == "blend" or d.meta_model is not None
