"""Feature pipeline: TF-IDF (word + char) fused with the heuristic signals."""

from __future__ import annotations

import re

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from ..core.features import HeuristicFeatures
from ..core.normalizer import normalise_text

_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_NUM = re.compile(r"\d+")


def normalise_for_tfidf(text: str) -> str:
    """
    Collapse the parts that carry no lexical signal.

    Homoglyph folding runs first, so `pаypal` (Cyrillic а) and `paypal` share
    a token instead of being two unrelated features — without it, script
    spoofing silently defeats the vectoriser. URLs then become one token
    (their *structure* is covered by the heuristic features) and long digit
    runs collapse, so the vectoriser cannot memorise a phone number or order
    id from the training corpus.
    """
    text = normalise_text(text or "").lower()
    text = _URL.sub(" urltoken ", text)
    return _NUM.sub(lambda m: " numtoken " if len(m.group()) > 3 else m.group(), text)


def build_features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(
                preprocessor=normalise_for_tfidf, ngram_range=(1, 2), min_df=2,
                max_df=0.9, sublinear_tf=True, strip_accents="unicode")),
            # char n-grams survive leetspeak, typos and word-splitting evasion
            ("char", TfidfVectorizer(
                preprocessor=normalise_for_tfidf, analyzer="char_wb",
                ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
            ("heuristic", Pipeline([("extract", HeuristicFeatures()),
                                    ("scale", MaxAbsScaler())])),
        ],
        transformer_weights={"word": 1.0, "char": 0.7, "heuristic": 1.4},
    )


def build_dense_features(word_components: int = 220, char_components: int = 90) -> FeatureUnion:
    """
    Dense projection for tree models.

    Gradient boosting cannot use a 100k-column sparse TF-IDF matrix, so the
    text blocks are reduced with truncated SVD (latent semantic analysis)
    while the heuristic features pass through unchanged - they are already
    dense, low-dimensional and individually meaningful, which is exactly what
    a tree splits on well.
    """
    return FeatureUnion([
        ("lsa_word", Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalise_for_tfidf, ngram_range=(1, 2),
                                      min_df=3, sublinear_tf=True, strip_accents="unicode")),
            ("svd", TruncatedSVD(n_components=word_components, random_state=7)),
        ])),
        ("lsa_char", Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalise_for_tfidf, analyzer="char_wb",
                                      ngram_range=(3, 5), min_df=4, sublinear_tf=True)),
            ("svd", TruncatedSVD(n_components=char_components, random_state=7)),
        ])),
        ("heuristic", HeuristicFeatures()),
    ])


def build_classifier(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")
    if kind == "linearsvc":
        return CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"),
                                      method="sigmoid", cv=3)
    if kind == "cnb":
        return ComplementNB(alpha=0.3)
    if kind == "sgd":
        # modified_huber is the loss that still exposes predict_proba.
        return SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=3000,
                             class_weight="balanced", random_state=7)
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            class_weight="balanced", random_state=7)
    raise ValueError(f"unknown classifier: {kind}")


#: Tree models need the dense projection; linear models keep the sparse union.
DENSE_MODELS = {"hgb"}


def build_pipeline(kind: str = "logreg") -> Pipeline:
    features = build_dense_features() if kind in DENSE_MODELS else build_features()
    return Pipeline([("features", features), ("clf", build_classifier(kind))])


def build_category_pipeline() -> Pipeline:
    """Predicts *which* scam playbook a flagged message follows."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalise_for_tfidf, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, C=6.0, class_weight="balanced")),
    ])
