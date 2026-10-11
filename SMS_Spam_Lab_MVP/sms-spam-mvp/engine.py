"""SMS spam MVP engine.

Local English demonstration built on the UCI SMS Spam Collection
(DOI 10.24432/C5CC84, CC BY 4.0).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parent / "data" / "SMSSpamCollection"

VALID_LABELS = {"ham", "spam"}

MAX_SMS_CHARS = 5000
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # matches TfidfVectorizer default


def canonical_text(text: str) -> str:
    """Canonical key: casefold, collapse all whitespace to single spaces."""
    return " ".join(text.casefold().split())


def _row_id(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _frame_sha256(df: pd.DataFrame) -> str:
    payload = "\n".join(f"{i}\t{l}" for i, l in zip(df["id"], df["label"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_pipeline() -> Pipeline:
    """TF-IDF (word uni+bigrams, sublinear tf, min_df=1) + LogisticRegression.

    Hyperparameters are fixed by contract; no test-dependent tuning.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(C=3, max_iter=1000, random_state=42)),
    ])


def _validate_two_classes(labels, context: str) -> None:
    unique = set(labels)
    if not unique <= VALID_LABELS:
        raise ValueError(
            f"{context}: labels must be a subset of {{'ham','spam'}}, got {sorted(unique)}")
    if unique != VALID_LABELS:
        raise ValueError(
            f"{context}: need both classes 'ham' and 'spam', got {sorted(unique) or '[]'}")


def fit_model(frame: pd.DataFrame) -> Pipeline:
    """Fit the fixed pipeline on a DataFrame with label/text columns."""
    for col in ("label", "text"):
        if col not in frame.columns:
            raise ValueError(f"fit_model: frame missing required column '{col}'")
    if len(frame) == 0:
        raise ValueError("fit_model: empty training frame")
    _validate_two_classes(frame["label"], "fit_model")
    pipe = make_pipeline()
    pipe.fit(frame["text"], frame["label"])
    return pipe


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict:
    """Evaluate on a held-out frame; returns the contract metrics dict.

    confusion_matrix layout: [[TN, FP], [FN, TP]] with spam as positive.
    """
    for col in ("label", "text"):
        if col not in test.columns:
            raise ValueError(f"evaluate: test frame missing required column '{col}'")
    _validate_two_classes(test["label"], "evaluate")
    y_true = test["label"].tolist()
    y_pred = model.predict(test["text"]).tolist()
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "ham" and p == "ham")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "ham" and p == "spam")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "spam" and p == "ham")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "spam" and p == "spam")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total = tn + fp + fn + tp
    return {
        "spam_precision": precision,
        "spam_recall": recall,
        "spam_f1": f1,
        "accuracy": (tn + tp) / total,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "test_rows": total,
    }


def _vocab_terms(model: Pipeline) -> set[str]:
    vocab = model.named_steps["tfidf"].vocabulary_
    return set(vocab.keys())


def predict_sms(model: Pipeline, text: str) -> dict:
    """Predict one SMS. Rejects blank text and text > 5000 chars.

    known_terms: input unigram tokens present in the fitted TF-IDF vocabulary.
    warning: None, or a string when the message contains no in-vocabulary terms.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("predict_sms: text must be a non-empty string")
    if len(text) > MAX_SMS_CHARS:
        raise ValueError(
            f"predict_sms: text exceeds {MAX_SMS_CHARS} characters (got {len(text)})")
    prob = float(model.predict_proba([text])[0, model.classes_.tolist().index("spam")])
    label = "spam" if prob >= 0.5 else "ham"
    tokens = [t.casefold() for t in _TOKEN_RE.findall(text)]
    vocab = _vocab_terms(model)
    known = sorted({t for t in tokens if t in vocab})
    warning = None
    if not known:
        warning = ("No terms from this message appear in the training vocabulary; "
                   "prediction is unreliable (out-of-vocabulary input).")
    return {
        "label": label,
        "spam_probability": prob,
        "known_terms": known,
        "warning": warning,
    }


def poison_training(untrusted: pd.DataFrame, rate: float = 0.60,
                    seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    """Label-flipping attack on a COPY of the untrusted pool.

    Flips a seeded fraction (floor(n_spam * rate)) of untrusted spam labels
    to ham. Returns (poisoned_frame, list_of_changed_ids). The input frame
    is never mutated. Rate must be in [0, 1].
    """
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"poison_training: rate must be in [0, 1], got {rate}")
    out = untrusted.copy().reset_index(drop=True)
    spam_idx = out.index[out["label"] == "spam"].tolist()
    n_flip = int(len(spam_idx) * rate)  # floor
    if n_flip == 0:
        return out, []
    rng = numpy.random.default_rng(seed)
    chosen = rng.choice(spam_idx, size=n_flip, replace=False)
    chosen = sorted(int(i) for i in chosen)
    out.loc[chosen, "label"] = "ham"
    changed_ids = out.loc[chosen, "id"].tolist()
    return out, changed_ids


def _reference_pipeline() -> Pipeline:
    """Reference model pipeline: same features, class_weight='balanced'
    (used ONLY by the defence reference model, never by experiment models)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(
            C=3, max_iter=1000, random_state=42, class_weight="balanced")),
    ])


def filter_suspicious(trusted: pd.DataFrame, untrusted: pd.DataFrame,
                      threshold: float = 0.65
                      ) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """Defence: quarantine suspicious 'ham' rows using a trusted reference.

    Fits a reference model on the trusted set only (class_weight='balanced'),
    then flags untrusted rows whose OBSERVED label is ham but whose reference
    spam probability >= threshold. Uses no clean labels and no attack
    manifest. Fixed threshold; may wrongly remove genuine ham rows.

    Returns (retained_untrusted, quarantined, reference_model). The
    quarantined frame carries a reference_spam_probability column.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"filter_suspicious: threshold must be in [0, 1], got {threshold}")
    _validate_two_classes(trusted["label"], "filter_suspicious(trusted)")
    untrusted = untrusted.copy().reset_index(drop=True)

    reference_model = _reference_pipeline()
    reference_model.fit(trusted["text"], trusted["label"])
    spam_col = reference_model.classes_.tolist().index("spam")
    probs = reference_model.predict_proba(untrusted["text"])[:, spam_col]
    untrusted["reference_spam_probability"] = probs

    observed_ham = untrusted["label"] == "ham"
    suspicious = observed_ham & (untrusted["reference_spam_probability"] >= threshold)

    quarantined = untrusted[suspicious].reset_index(drop=True)
    retained = untrusted[~suspicious].drop(
        columns=["reference_spam_probability"]).reset_index(drop=True)
    return retained, quarantined, reference_model


def prepare_data(path: Path | str = DATA_PATH, seed: int = 42) -> dict:
    """Parse, clean and split the corpus into frozen train/test partitions.

    Cleaning rules:
    - split each line on the FIRST tab only; label must be ham/spam and text
      non-empty; malformed rows are counted in stats['malformed_rows']
      (never silently dropped without provenance);
    - canonical text key = ' '.join(text.casefold().split());
    - canonical keys carrying conflicting labels are removed entirely;
    - remaining duplicate keys are deduplicated BEFORE the split.

    Raises ValueError if the cleaned corpus does not contain both classes
    with enough rows for a stratified 80/20 split (>= 2 rows per class).

    Split: stratified 80/20 with the given seed (default 42), frozen.
    Trusted = stratified 25% of TRAIN only; untrusted = the rest of train.
    """
    path = Path(path)
    corpus_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    raw_rows = []
    raw_ham = raw_spam = malformed_rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue  # blank separator line, not a data row
        if "\t" not in line:
            malformed_rows += 1
            continue
        label, text = line.split("\t", 1)
        label = label.strip()
        if label not in VALID_LABELS:
            malformed_rows += 1
            continue
        if not text.strip():
            malformed_rows += 1
            continue
        raw_rows.append((label, text))
        if label == "ham":
            raw_ham += 1
        else:
            raw_spam += 1

    # Group by canonical key; drop keys with conflicting labels.
    by_key: dict[str, list[str]] = {}
    labels_by_key: dict[str, set[str]] = {}
    for label, text in raw_rows:
        key = canonical_text(text)
        by_key.setdefault(key, []).append(text)
        labels_by_key.setdefault(key, set()).add(label)

    conflicting_keys = {k for k, ls in labels_by_key.items() if len(ls) > 1}
    conflicts_removed = sum(len(by_key[k]) for k in conflicting_keys)

    records = []
    for key, texts in by_key.items():
        if key in conflicting_keys:
            continue
        label = next(iter(labels_by_key[key]))
        records.append({"id": _row_id(key), "label": label, "text": texts[0]})
    clean = pd.DataFrame(records, columns=["id", "label", "text"])
    clean = clean.sort_values("id").reset_index(drop=True)

    duplicates_removed = len(raw_rows) - conflicts_removed - len(clean)
    clean_ham = int((clean["label"] == "ham").sum())
    clean_spam = int((clean["label"] == "spam").sum())

    if clean_ham < 2 or clean_spam < 2:
        raise ValueError(
            "prepare_data: cleaned corpus has insufficient classes for a "
            f"stratified 80/20 split (ham={clean_ham}, spam={clean_spam}; "
            "need >= 2 of each)")

    train, test = train_test_split(
        clean, test_size=0.20, random_state=seed, stratify=clean["label"])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    trusted, untrusted = train_test_split(
        train, test_size=0.75, random_state=seed, stratify=train["label"])
    trusted = trusted.reset_index(drop=True)
    untrusted = untrusted.reset_index(drop=True)

    stats = {
        "raw_rows": len(raw_rows),
        "malformed_rows": malformed_rows,
        "unique_rows": len(clean),
        "duplicates_removed": duplicates_removed,
        "conflicts_removed": conflicts_removed,
        "train_rows": len(train),
        "test_rows": len(test),
        "trusted_rows": len(trusted),
        "untrusted_rows": len(untrusted),
        "raw_ham": raw_ham,
        "raw_spam": raw_spam,
        "clean_ham": clean_ham,
        "clean_spam": clean_spam,
        "corpus_sha256": corpus_sha256,
    }
    return {
        "train": train,
        "test": test,
        "trusted": trusted,
        "untrusted": untrusted,
        "stats": stats,
        "seed": seed,
        "test_fingerprint": _frame_sha256(test),
    }


def run_experiment(prepared: dict, poison_rate: float = 0.60,
                   defence_threshold: float = 0.65, seed: int = 42) -> dict:
    trusted_ids = set(prepared['trusted']['id'])
    untrusted_ids = set(prepared['untrusted']['id'])
    test_ids = set(prepared['test']['id'])
    if (trusted_ids & untrusted_ids or test_ids & (trusted_ids | untrusted_ids)
            or trusted_ids | untrusted_ids != set(prepared['train']['id'])):
        raise ValueError('Training/test partition overlap or inconsistent training partitions')
    poisoned, manifest = poison_training(prepared['untrusted'], rate=poison_rate, seed=seed)
    retained, quarantined, reference = filter_suspicious(
        prepared['trusted'], poisoned, threshold=defence_threshold)
    poisoned_train = pd.concat([prepared['trusted'], poisoned], ignore_index=True)
    defended_train = pd.concat([prepared['trusted'], retained], ignore_index=True)
    models = {
        'Baseline': fit_model(prepared['train']),
        'Poisoned': fit_model(poisoned_train),
        'Defended': fit_model(defended_train),
    }
    return {
        'models': models,
        'metrics': {name: evaluate(model, prepared['test']) for name, model in models.items()},
        'summary': {
            'poison_rate': float(poison_rate), 'defence_threshold': float(defence_threshold),
            'changed_rows': len(manifest), 'quarantined_rows': len(quarantined),
            'trusted_rows': len(prepared['trusted']), 'untrusted_rows': len(prepared['untrusted']),
            'defended_training_rows': len(defended_train),
            'changed_fraction_of_all_training_rows': len(manifest) / len(prepared['train']),
            'seed': seed, 'data_split_seed': prepared['seed'],
            'test_fingerprint': prepared['test_fingerprint'],
        },
        'poisoned_train': poisoned_train, 'quarantined': quarantined,
        'attack_manifest': manifest, 'reference_model': reference,
    }


def report_for_json(result: dict, prepared: dict) -> dict:
    from datetime import datetime
    from importlib.metadata import version
    return {
        'format_version': 1,
        'generated_at': datetime.now().astimezone().isoformat(),
        'model': 'TF-IDF word unigram/bigram + Logistic Regression',
        'metrics': result['metrics'], 'summary': result['summary'], 'data': prepared['stats'],
        'environment': {name: version(name) for name in ['scikit-learn', 'numpy', 'pandas', 'joblib']},
        'assumptions': [
            'Protected correctly labelled reference subset: 25 percent of TRAIN only.',
            'Attacker flips observed spam labels in untrusted training only; no text changes.',
            'Defence receives no clean labels of attacked rows, attack manifest, or test rows.',
            'Same clean holdout; default seed 42; thresholds fixed before the default run.',
        ],
        'limitations': [
            'One old largely English spam/ham corpus, not phishing labels or modern-scam coverage.',
            'One exploratory split, not a validated robustness or deployment claim.',
            'Quarantine can remove genuine ham; protection depends on the trusted-set assumption.',
            'Repeated tuning against this visible test set requires a fresh final holdout.',
        ],
    }
