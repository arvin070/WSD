"""
Learned meta-calibrator.

The detector previously fused the text model and the rule engine with a fixed
weighted average (`0.85 * ml + 0.15 * rules`) whose weight was grid-searched
on the dev split. That treats the two scores as interchangeable evidence at
every point in the space, which they are not: the rule score is informative
when it fires on a short unambiguous message and close to noise on long prose,
and the right weighting depends on context the blend cannot see.

This replaces it with a small logistic regression stacked on top of both
scores plus a handful of context features. It is deliberately tiny — eleven
inputs — because the meta-model is fitted on out-of-fold predictions from a
corpus with only a few hundred template groups, and anything larger would
memorise them.

**The out-of-fold requirement is not optional.** Training the meta-model on
in-sample base predictions would teach it that `ml_score` is near-perfect,
because on training rows it is. It must see the base model's honest
generalisation error, which is what `cross_val_predict` provides.
"""

from __future__ import annotations

import numpy as np

from ..core.features import analyse_message
from ..core.features import rule_score as compute_rule_score

#: Context features the blend could not use. Kept short on purpose.
META_CONTEXT_FEATURES = [
    "has_ask",
    "n_narrative",
    "x_inquiry_not_directive",
    "len_words",
    "has_url",
    "has_lookalike",
    "n_homoglyphs",
    "char_entropy",
    "n_zero_width",
]
META_FEATURE_NAMES = ["ml_score", "rule_score", *META_CONTEXT_FEATURES]


def meta_features(texts: list[str], ml_scores: np.ndarray,
                  rule_scores: np.ndarray | None = None) -> np.ndarray:
    """Build the meta-model design matrix for `texts` given base ML scores."""
    ml_scores = np.asarray(ml_scores, dtype=np.float64)
    if rule_scores is None:
        rule_scores = np.array([compute_rule_score(t)[0] for t in texts])
    rule_scores = np.asarray(rule_scores, dtype=np.float64)

    rows = []
    for text, ml, rule in zip(texts, ml_scores, rule_scores, strict=True):
        feats = analyse_message(text)[0]
        rows.append([ml, rule, *(feats[name] for name in META_CONTEXT_FEATURES)])
    return np.asarray(rows, dtype=np.float64)


def build_meta_model():
    """Regularised logistic regression - small data, so keep it constrained."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")


def meta_predict(model, texts: list[str], ml_scores: np.ndarray,
                 rule_scores: np.ndarray | None = None) -> np.ndarray:
    X = meta_features(texts, ml_scores, rule_scores)
    return model.predict_proba(X)[:, 1]
