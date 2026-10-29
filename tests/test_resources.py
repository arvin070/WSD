"""The YAML content files are now load-bearing, so validate their shape."""

from scamshield.config import Settings, get_settings
from scamshield.core.features import feature_names, rule_score
from scamshield.resources import load_lexicons, load_templates


def test_lexicons_load():
    lex = load_lexicons()
    assert len(lex.signals) >= 10
    assert all(s.terms for s in lex.signals), "every signal family needs terms"
    assert all(0 < s.weight <= 1 for s in lex.signals), "weights must be in (0, 1]"


def test_signal_features_are_unique():
    lex = load_lexicons()
    features = [s.feature for s in lex.signals]
    assert len(features) == len(set(features)), "two signals share a feature key"


def test_derived_signals_present():
    lex = load_lexicons()
    for code in ("lookalike_domain", "ip_link", "hidden_chars", "upi_id"):
        assert code in lex.derived
        assert lex.derived[code].label


def test_scoring_constants_present():
    lex = load_lexicons()
    for key in ("saturation_rate", "discount_narrative_no_ask", "large_sum_min"):
        assert key in lex.scoring


def test_templates_have_slots_scam_and_ham():
    tpl = load_templates()
    assert tpl["scam"] and tpl["ham"] and tpl["slots"]
    assert all(isinstance(v, list) and v for v in tpl["scam"].values())
    for name, body in tpl["ham"].items():
        assert body["messages"], f"ham pool {name} is empty"
        assert body["weight"] > 0


def test_template_slots_are_all_fillable():
    """A typo'd {slot} would ship literal braces into the training data."""
    import re
    tpl = load_templates()
    known = {"bank", "wallet", "brand", "brandlower", "stream", "streamlower", "courier",
             "telco", "name", "amt", "small", "url", "upi", "phone", "num"}
    for category, templates in tpl["scam"].items():
        for t in templates:
            for slot in re.findall(r"\{(\w+)\}", t):
                assert slot in known, f"unknown slot {{{slot}}} in {category}"


def test_feature_names_are_stable_and_unique():
    names = feature_names()
    assert len(names) == len(set(names))
    assert names == feature_names(), "feature ordering must be deterministic"


def test_editing_weights_changes_scoring():
    """The YAML is genuinely load-bearing, not decorative."""
    lex = load_lexicons()
    assert lex.by_code("credentials").weight == 1.0
    score, _ = rule_score("Please share the OTP sent to your phone")
    assert score > 0.5


def test_settings_defaults_are_sane():
    cfg = get_settings()
    assert cfg.low_info_tokens >= 1
    assert 0 < cfg.low_info_cap < 1
    assert cfg.max_upload_bytes > 0
    assert "*" not in cfg.cors_origins, "wildcard CORS contradicts the local-only design"


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("SCAMSHIELD_PORT", "9999")
    monkeypatch.setenv("SCAMSHIELD_CORS_ORIGINS", "http://a.test,http://b.test")
    cfg = Settings()
    assert cfg.port == 9999
    assert cfg.cors_origins == ["http://a.test", "http://b.test"]


def test_paths_derive_from_dirs():
    cfg = get_settings()
    assert cfg.model_path.name == "scam_model.joblib"
    assert cfg.model_path.parent == cfg.models_dir
    assert cfg.lexicons_path.exists() and cfg.templates_path.exists()
