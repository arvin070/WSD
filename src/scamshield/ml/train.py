"""
Train, evaluate and persist the scam detector.

    scamshield-train          (or: python -m scamshield.ml.train)

Methodology — why the reported numbers mean something:
  * Corpus splits are **grouped by template id**, so an augmented variant of a
    training message can never leak into a validation fold.
  * Model choice, the ML/rule blend weight and the decision threshold are all
    tuned on `data/realworld.jsonl` split=dev — hand-written messages sharing
    no templates with the generator.
  * Headline metrics come from split=test, never read for any decision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from ..config import get_settings
from ..core.features import rule_score
from ..logging_setup import configure_logging, get_logger
from .calibrator import META_FEATURE_NAMES, build_meta_model, meta_features, meta_predict
from .corpus import build_corpus
from .pipeline import build_category_pipeline, build_pipeline

log = get_logger(__name__)

CANDIDATES = ["logreg", "linearsvc", "cnb", "sgd", "hgb"]
#: ComplementNB stays as a cross-validation baseline but is excluded from
#: selection: its posteriors are not usable as a calibrated risk score. It
#: saturates to 1.000 on short conversational text ("I want to withdraw it"),
#: which matters because the score is shown to users as a percentage.
SELECTABLE = ["logreg", "linearsvc", "sgd", "hgb", "ensemble"]


def load_external_holdout() -> list[dict]:
    """
    Third-party evaluation split, held out by template signature.

    Two orders of magnitude larger than the hand-written test set and written
    by someone else, so it checks the model against vocabulary and phrasing
    this project did not invent.
    """
    path = get_settings().data_dir / "external_holdout.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_realworld() -> tuple[list[dict], list[dict]]:
    path = get_settings().data_dir / "realworld.jsonl"
    if not path.exists():
        return [], []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return ([r for r in rows if r.get("split") == "dev"],
            [r for r in rows if r.get("split") == "test"])


def metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float((tp + tn) / max(tp + tn + fp + fn, 1)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": round(float(threshold), 3),
    }


def blend(p_ml, p_rule, alpha: float) -> np.ndarray:
    """Weighted fusion. ML stays dominant; rules add lift for unseen wording."""
    return np.clip((1 - alpha) * np.asarray(p_ml) + alpha * np.asarray(p_rule), 0.0, 1.0)


def _short(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "confusion"}


def train(output_dir: Path | None = None) -> dict:
    settings = get_settings()
    out = output_dir or settings.models_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log.info("[1/7] building corpus")
    df = pd.DataFrame(build_corpus())
    X, y = df["text"].tolist(), df["label"].to_numpy()
    groups = df["template_id"].to_numpy()
    log.info("      %d rows | scam=%d ham=%d | %d template groups",
             len(df), int(y.sum()), int((1 - y).sum()), df["template_id"].nunique())

    dev, test = load_realworld()
    if not dev or not test:
        raise SystemExit("data/realworld.jsonl with dev+test splits is required")
    dx, dy = [r["text"] for r in dev], np.array([r["label"] for r in dev])
    tx, ty = [r["text"] for r in test], np.array([r["label"] for r in test])
    log.info("      dev=%d test=%d (hand-written, disjoint from templates)", len(dev), len(test))

    log.info("[2/7] grouped cross-validation (sanity check on the synthetic corpus)")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    cv_scores, oof = {}, {}
    for kind in CANDIDATES:
        probs = cross_val_predict(build_pipeline(kind), X, y, groups=groups, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
        oof[kind] = probs
        cv_scores[kind] = metrics(y, probs, 0.5)
        log.info("      %-10s f1=%.4f pr_auc=%.4f", kind, cv_scores[kind]["f1"], cv_scores[kind]["pr_auc"])
    oof["ensemble"] = (oof["logreg"] + oof["linearsvc"]) / 2

    log.info("[3/7] fitting candidates on the full corpus, scoring on dev")
    fitted = {k: build_pipeline(k).fit(X, y) for k in CANDIDATES}

    def predict_ml(name: str, texts: list[str]) -> np.ndarray:
        if name == "ensemble":
            return np.mean([fitted[k].predict_proba(texts)[:, 1] for k in ("logreg", "linearsvc")], axis=0)
        return fitted[name].predict_proba(texts)[:, 1]

    rule_dev = np.array([rule_score(t)[0] for t in dx])
    rule_test = np.array([rule_score(t)[0] for t in tx])

    dev_scores = {n: metrics(dy, predict_ml(n, dx), 0.5) for n in [*CANDIDATES, "ensemble"]}
    for n, m in dev_scores.items():
        log.info("      %-10s dev pr_auc=%.4f roc_auc=%.4f", n, m["pr_auc"], m["roc_auc"])

    log.info("[4/7] selecting model + blend weight + threshold on dev")
    per_model = {}
    for name in SELECTABLE:
        p_ml = predict_ml(name, dx)
        grid = []
        for alpha in np.arange(0.0, 0.55, 0.05):
            blended = blend(p_ml, rule_dev, float(alpha))
            for thr in np.arange(0.15, 0.85, 0.025):
                # F-beta with beta=2: a missed scam costs the user money, a
                # false alarm costs them a moment's doubt.
                grid.append((fbeta_score(dy, (blended >= thr).astype(int), beta=2,
                                         zero_division=0), float(alpha), float(thr)))
        best = max(g[0] for g in grid)
        # Among settings tied on dev, prefer a small positive blend weight:
        # the rule engine is the only component that reacts to wording absent
        # from the corpus, so it is free robustness at zero measured cost.
        tied = [g for g in grid if g[0] >= best - 1e-9]
        # Invariant: the rule engine's maximum contribution must stay below
        # the decision threshold. Otherwise a single rule hit crosses the line
        # on its own, and the rules' standalone precision (~0.76) becomes the
        # system's precision ceiling regardless of how good the model is.
        safe = [g for g in tied if g[1] < g[2]] or tied
        score, alpha, thr = min(safe, key=lambda g: (abs(g[1] - 0.15), g[2]))
        per_model[name] = {
            "score": score, "alpha": alpha, "threshold": thr,
            "brier": float(brier_score_loss(dy, blend(p_ml, rule_dev, alpha))),
            "f1": float(f1_score(dy, (blend(p_ml, rule_dev, alpha) >= thr).astype(int), zero_division=0)),
        }
        log.info("      %-10s dev f2=%.4f f1=%.4f brier=%.4f alpha=%.2f thr=%.3f",
                 name, score, per_model[name]["f1"], per_model[name]["brier"], alpha, thr)

    best_f2 = max(v["score"] for v in per_model.values())
    shortlist = {k: v for k, v in per_model.items() if v["score"] >= best_f2 - 0.02}
    name = min(shortlist, key=lambda k: shortlist[k]["brier"])
    alpha, threshold = per_model[name]["alpha"], per_model[name]["threshold"]
    log.info("      -> model=%s alpha=%.2f threshold=%.3f (dev f2=%.4f, brier=%.4f)",
             name, alpha, threshold, per_model[name]["score"], per_model[name]["brier"])

    log.info("[4b/7] fitting the learned meta-calibrator on out-of-fold predictions")
    # Out-of-fold, not in-sample: on training rows the base model is nearly
    # perfect, and a meta-model fitted on that would learn to trust ml_score
    # unconditionally. cross_val_predict gives it the honest error instead.
    rule_train = np.array([rule_score(t)[0] for t in X])
    meta_model = build_meta_model().fit(meta_features(X, oof[name], rule_train), y)

    dev_meta = meta_predict(meta_model, dx, predict_ml(name, dx), rule_dev)
    meta_best = max(
        ((fbeta_score(dy, (dev_meta >= t).astype(int), beta=2, zero_division=0), float(t))
         for t in np.arange(0.15, 0.90, 0.025)), key=lambda g: (g[0], -g[1]))
    blend_f2 = per_model[name]["score"]
    log.info("      meta  dev f2=%.4f (thr %.3f) | fixed blend dev f2=%.4f (alpha %.2f)",
             meta_best[0], meta_best[1], blend_f2, alpha)

    coefs = dict(zip(META_FEATURE_NAMES, meta_model.coef_[0].round(3), strict=True))
    log.info("      meta weights: %s", coefs)

    # Only adopt the learned fusion if it actually beats the fixed blend on
    # dev. A more sophisticated model that measures worse is not an upgrade.
    use_meta = meta_best[0] > blend_f2 + 1e-9
    fusion = "meta" if use_meta else "blend"
    if use_meta:
        threshold = meta_best[1]
    log.info("      -> fusion=%s", fusion)

    # A higher band for "this is definitely a scam" messaging. Precision is
    # already 1.0 at the base threshold, so searching upward for a precision
    # target returns the base threshold and the band collapses. Anchor it to
    # where confident positives actually sit instead.
    blended_dev = dev_meta if use_meta else blend(predict_ml(name, dx), rule_dev, alpha)
    pos = blended_dev[dy == 1]
    strict = float(np.clip(max(threshold + 0.15, float(np.percentile(pos, 25))),
                           threshold + 0.15, 0.95))
    log.info("      high-confidence band starts at %.3f (covers %.0f%% of dev scams)",
             strict, float((pos >= strict).mean()) * 100)

    log.info("[5/7] training the scam-category model")
    # External corpora carry no playbook label - their "category" is just the
    # source name. Training on those would have the model answer "this is a
    # uci_sms_spam" when a user asks what kind of scam they are looking at.
    scam_df = df[(df["label"] == 1) & (~df["category"].str.startswith("external:"))]
    cats = scam_df["category"].tolist()
    cat_model = build_category_pipeline().fit(scam_df["text"].tolist(), cats)
    cat_cv_args = {"groups": scam_df["template_id"].to_numpy(),
                   "cv": StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7),
                   "n_jobs": -1}
    cat_pred = cross_val_predict(build_category_pipeline(), scam_df["text"].tolist(), cats, **cat_cv_args)
    cat_proba = cross_val_predict(build_category_pipeline(), scam_df["text"].tolist(), cats,
                                  method="predict_proba", **cat_cv_args)
    cat_acc = float((cat_pred == scam_df["category"].to_numpy()).mean())
    classes = np.array(sorted(set(cats)))
    top2 = classes[np.argsort(-cat_proba, axis=1)[:, :2]]
    truth = scam_df["category"].to_numpy()
    cat_top2 = float(np.mean([truth[i] in top2[i] for i in range(len(top2))]))
    log.info("      grouped-CV accuracy: %.4f (top-2: %.4f) over %d categories",
             cat_acc, cat_top2, len(classes))

    log.info("[6/7] final evaluation on the held-out test splits")
    p_ml_t = predict_ml(name, tx)
    p_blend_t = blend(p_ml_t, rule_test, alpha)
    p_meta_t = meta_predict(meta_model, tx, p_ml_t, rule_test)
    p_final = p_meta_t if use_meta else p_blend_t
    report = {
        "blended": metrics(ty, p_final, threshold),
        "meta_fusion": metrics(ty, p_meta_t, threshold),
        "fixed_blend": metrics(ty, p_blend_t, per_model[name]["threshold"]),
        "ml_only": metrics(ty, p_ml_t, threshold),
        "rules_only": metrics(ty, rule_test, 0.5),
        "dev_blended": metrics(dy, blended_dev, threshold),
    }
    log.info("      blended   : %s", _short(report["blended"]))
    log.info("      ml only   : %s", _short(report["ml_only"]))
    log.info("      rules only: %s", _short(report["rules_only"]))
    print()
    print(classification_report(ty, (p_final >= threshold).astype(int),
                                target_names=["legitimate", "scam"], zero_division=0))
    errs = [(tx[i][:88], round(float(p_final[i]), 3), int(ty[i]))
            for i in range(len(ty)) if int(p_final[i] >= threshold) != ty[i]]
    for t, sc, lab in errs:
        log.info("      [%5s] %s %s", sc, "MISSED SCAM" if lab == 1 else "FALSE ALARM", t)

    external = load_external_holdout()
    if external:
        ex = [r["text"] for r in external]
        ey = np.array([r["label"] for r in external])
        p_ext = blend(predict_ml(name, ex), np.array([rule_score(t)[0] for t in ex]), alpha)
        report["external_holdout"] = metrics(ey, p_ext, threshold)
        log.info("      external holdout (%d msgs, template-disjoint): %s",
                 len(external), _short(report["external_holdout"]))

    log.info("[7/7] saving artifacts")
    members = [fitted["logreg"], fitted["linearsvc"]] if name == "ensemble" else [fitted[name]]
    joblib.dump({"members": members, "kind": name, "category_model": cat_model,
                 "alpha": alpha, "threshold": threshold, "strict_threshold": strict,
                 "meta_model": meta_model, "fusion": fusion},
                out / "scam_model.joblib", compress=3)
    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_size": int(len(df)), "scam_rows": int(y.sum()), "ham_rows": int((1 - y).sum()),
        "template_groups": int(df["template_id"].nunique()),
        "selected_model": name, "blend_alpha": round(alpha, 3),
        "fusion": fusion,
        "meta_weights": {k: float(v) for k, v in coefs.items()},
        "meta_dev_f2": round(float(meta_best[0]), 4),
        "blend_dev_f2": round(float(blend_f2), 4),
        "threshold": round(threshold, 3), "strict_threshold": round(strict, 3),
        "categories": sorted(set(cats)),
        "cv_grouped_synthetic": cv_scores, "dev_model_selection": dev_scores,
        "dev_tuning": per_model, "holdout_test": report,
        "category_cv_accuracy": round(cat_acc, 4),
        "category_cv_top2_accuracy": round(cat_top2, 4),
        "train_seconds": round(time.time() - t0, 1),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.data_dir / "corpus.csv", index=False)
    log.info("      -> %s", out / "scam_model.joblib")
    log.info("done in %.1fs", time.time() - t0)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ScamShield detector.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to write scam_model.joblib (default: models/)")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    train(args.output_dir)


if __name__ == "__main__":
    main()
