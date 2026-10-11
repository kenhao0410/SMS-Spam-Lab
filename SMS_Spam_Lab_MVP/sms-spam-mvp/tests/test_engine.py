"""TDD tests for the SMS spam MVP engine (UCI SMSSpamCollection)."""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402


@pytest.fixture(scope="session")
def prepared():
    return engine.prepare_data()


# ---------------------------------------------------------------- slice 1
class TestPrepareData:
    def test_returns_required_keys(self, prepared):
        for key in ("train", "test", "trusted", "untrusted", "stats",
                    "seed", "test_fingerprint"):
            assert key in prepared

    def test_frames_have_id_label_text(self, prepared):
        for name in ("train", "test", "trusted", "untrusted"):
            df = prepared[name]
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["id", "label", "text"]
            assert set(df["label"].unique()) <= {"ham", "spam"}
            assert (df["text"].str.strip() != "").all()
            # id is sha256 hex of canonical text
            for _, row in df.head(20).iterrows():
                canonical = " ".join(row["text"].casefold().split())
                assert row["id"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def test_split_80_20_seed42(self, prepared):
        train, test = prepared["train"], prepared["test"]
        stats = prepared["stats"]
        total = len(train) + len(test)
        assert stats["train_rows"] == len(train)
        assert stats["test_rows"] == len(test)
        assert stats["unique_rows"] == total
        assert abs(len(test) / total - 0.20) < 0.01
        assert prepared["seed"] == 42

    def test_train_test_disjoint(self, prepared):
        assert set(prepared["train"]["id"]) & set(prepared["test"]["id"]) == set()

    def test_trusted_untrusted_disjoint_subsets_of_train(self, prepared):
        train_ids = set(prepared["train"]["id"])
        trusted, untrusted = prepared["trusted"], prepared["untrusted"]
        assert set(trusted["id"]) & set(untrusted["id"]) == set()
        assert set(trusted["id"]) <= train_ids
        assert set(untrusted["id"]) <= train_ids
        # trusted is 25% of train, trusted+untrusted == train
        assert len(trusted) == pytest.approx(0.25 * len(prepared["train"]), abs=1)
        assert len(trusted) + len(untrusted) == len(prepared["train"])

    def test_stats_keys_and_consistency(self, prepared):
        stats = prepared["stats"]
        for key in ("raw_rows", "unique_rows", "duplicates_removed",
                    "conflicts_removed", "train_rows", "test_rows",
                    "trusted_rows", "untrusted_rows", "raw_ham", "raw_spam",
                    "clean_ham", "clean_spam", "corpus_sha256"):
            assert key in stats, key
        assert stats["raw_rows"] == stats["raw_ham"] + stats["raw_spam"]
        assert (stats["unique_rows"] ==
                stats["raw_rows"] - stats["duplicates_removed"] - stats["conflicts_removed"])
        assert stats["clean_ham"] + stats["clean_spam"] == stats["unique_rows"]
        assert stats["corpus_sha256"] == (
            "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d")

    def test_deterministic(self, prepared):
        again = engine.prepare_data()
        assert again["test_fingerprint"] == prepared["test_fingerprint"]
        assert again["stats"] == prepared["stats"]
        pd.testing.assert_frame_equal(again["train"], prepared["train"])
        pd.testing.assert_frame_equal(again["test"], prepared["test"])

    def test_test_fingerprint_matches_test_frame(self, prepared):
        test = prepared["test"]
        payload = "\n".join(f"{i}\t{l}" for i, l in
                            zip(test["id"], test["label"]))
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert prepared["test_fingerprint"] == expected

    def test_malformed_and_empty_lines_skipped(self, tmp_path):
        p = tmp_path / "corpus.tsv"
        good = [f"ham\tGood message number {i}\n" for i in range(12)]
        good += [f"spam\tBuy cheap stuff number {i}\n" for i in range(12)]
        bad = ("ham\n"                      # missing text
               "junk\tBad label row\n"      # invalid label
               "\tNo label here\n"
               "ham\t   \n")               # whitespace-only text
        p.write_text("".join(good) + bad, encoding="utf-8")
        out = engine.prepare_data(path=p, seed=42)
        assert out["stats"]["raw_rows"] == 24
        assert out["stats"]["unique_rows"] == 24
        # malformed rows are counted, never silently dropped without provenance
        assert out["stats"]["malformed_rows"] == 4

    def test_real_corpus_has_zero_malformed_rows(self, prepared):
        assert prepared["stats"]["malformed_rows"] == 0

    def test_insufficient_classes_raises_valueerror(self, tmp_path):
        p = tmp_path / "corpus.tsv"
        rows = [f"ham\tOnly ham message {i}\n" for i in range(10)]
        rows += ["spam\tSingle spam message\n"]
        p.write_text("".join(rows), encoding="utf-8")
        with pytest.raises(ValueError, match="insufficient classes"):
            engine.prepare_data(path=p, seed=42)

    def test_conflicting_label_keys_removed(self, tmp_path):
        p = tmp_path / "corpus.tsv"
        rows = [f"ham\tPlain ham message {i}\n" for i in range(12)]
        rows += [f"spam\tPlain spam message {i}\n" for i in range(12)]
        # same canonical text under both labels -> whole key removed
        rows += ["ham\tConflicting message text\n",
                 "spam\tconflicting   MESSAGE text\n"]
        p.write_text("".join(rows), encoding="utf-8")
        out = engine.prepare_data(path=p, seed=42)
        assert out["stats"]["conflicts_removed"] == 2
        assert out["stats"]["unique_rows"] == 24
        ids = set(pd.concat([out["train"], out["test"]])["id"])
        conflict_id = hashlib.sha256(
            "conflicting message text".encode("utf-8")).hexdigest()
        assert conflict_id not in ids

    def test_parser_splits_first_tab_only(self, tmp_path):
        p = tmp_path / "corpus.tsv"
        tabby = [f"ham\tTab text {i}\textra tabs\tinside\n" for i in range(12)]
        rows = tabby + [f"spam\tSpam number {i}\n" for i in range(12)]
        p.write_text("".join(rows), encoding="utf-8")
        out = engine.prepare_data(path=p, seed=42)
        texts = set(pd.concat([out["train"], out["test"]])["text"])
        assert "Tab text 0\textra tabs\tinside" in texts


# ---------------------------------------------------------------- slice 2
@pytest.fixture(scope="session")
def baseline(prepared):
    model = engine.fit_model(prepared["train"])
    return model


class TestModelAndEvaluate:
    def test_make_pipeline_is_sklearn_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        pipe = engine.make_pipeline()
        assert isinstance(pipe, Pipeline)
        tfidf = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
        assert isinstance(tfidf, TfidfVectorizer)
        assert tfidf.sublinear_tf is True
        assert tfidf.min_df == 1
        assert tfidf.ngram_range == (1, 2)
        assert isinstance(clf, LogisticRegression)
        assert clf.C == 3
        assert clf.max_iter == 1000
        assert clf.random_state == 42

    def test_fit_model_returns_pipeline(self, baseline):
        from sklearn.pipeline import Pipeline
        assert isinstance(baseline, Pipeline)

    def test_fit_model_rejects_single_class(self):
        df = pd.DataFrame({
            "id": ["a"] * 5,
            "label": ["ham"] * 5,
            "text": ["one msg", "two msg", "three msg", "four msg", "five msg"],
        })
        with pytest.raises(ValueError):
            engine.fit_model(df)

    def test_evaluate_keys_and_consistency(self, baseline, prepared):
        res = engine.evaluate(baseline, prepared["test"])
        for key in ("spam_precision", "spam_recall", "spam_f1", "accuracy",
                    "false_positives", "false_negatives", "confusion_matrix",
                    "test_rows"):
            assert key in res, key
        cm = res["confusion_matrix"]
        assert len(cm) == 2 and all(len(r) == 2 for r in cm)
        tn, fp = cm[0]
        fn, tp = cm[1]
        assert res["test_rows"] == len(prepared["test"])
        assert tn + fp + fn + tp == res["test_rows"]
        assert res["false_positives"] == fp
        assert res["false_negatives"] == fn
        assert res["spam_precision"] == pytest.approx(tp / (tp + fp), abs=1e-9)
        assert res["spam_recall"] == pytest.approx(tp / (tp + fn), abs=1e-9)
        assert res["accuracy"] == pytest.approx((tn + tp) / res["test_rows"], abs=1e-9)
        for k in ("spam_precision", "spam_recall", "spam_f1", "accuracy"):
            assert 0.0 <= res[k] <= 1.0

    def test_evaluate_rejects_single_class(self, baseline):
        df = pd.DataFrame({"id": ["x"], "label": ["ham"], "text": ["hello"]})
        with pytest.raises(ValueError):
            engine.evaluate(baseline, df)


class TestPredictSms:
    def test_predict_spam_message(self, baseline):
        out = engine.predict_sms(
            baseline,
            "URGENT! You have won a 1 week FREE membership! Call 09061701461 now!")
        assert out["label"] == "spam"
        assert 0.0 <= out["spam_probability"] <= 1.0

    def test_predict_ham_message(self, baseline):
        out = engine.predict_sms(baseline, "Hey, are you free for lunch tomorrow?")
        assert out["label"] == "ham"
        assert 0.0 <= out["spam_probability"] <= 1.0

    def test_predict_result_keys(self, baseline):
        out = engine.predict_sms(baseline, "Call me later")
        for key in ("label", "spam_probability", "known_terms", "warning"):
            assert key in out
        assert isinstance(out["known_terms"], list)

    def test_predict_known_terms_from_vocabulary(self, baseline):
        out = engine.predict_sms(baseline, "free entry to win tickets call now")
        known = set(out["known_terms"])
        assert {"free", "win", "call"} <= known  # common spam unigrams

    def test_predict_blank_rejected(self, baseline):
        with pytest.raises(ValueError):
            engine.predict_sms(baseline, "   ")
        with pytest.raises(ValueError):
            engine.predict_sms(baseline, "")

    def test_predict_too_long_rejected(self, baseline):
        with pytest.raises(ValueError):
            engine.predict_sms(baseline, "a" * 5001)

    def test_predict_oov_warning(self, baseline):
        out = engine.predict_sms(baseline, "zqxjvkw zzqqwplm xrblxnv")
        assert out["warning"] is not None
        assert isinstance(out["warning"], str)
        assert out["known_terms"] == []

    def test_predict_in_vocab_no_warning(self, baseline):
        out = engine.predict_sms(baseline, "call me later please")
        assert out["warning"] is None


# ---------------------------------------------------------------- slice 3
class TestPoisonTraining:
    def test_returns_copied_frame_and_changed_ids(self, prepared):
        untrusted = prepared["untrusted"]
        out, changed = engine.poison_training(untrusted, rate=0.60, seed=42)
        assert isinstance(out, pd.DataFrame)
        assert isinstance(changed, list)
        assert out is not untrusted
        # original untrusted frame untouched
        assert (untrusted["label"] == prepared["train"].set_index("id")
                .loc[untrusted["id"], "label"].values).all()

    def test_flips_only_spam_to_ham_at_seeded_rate(self, prepared):
        untrusted = prepared["untrusted"]
        n_spam = int((untrusted["label"] == "spam").sum())
        out, changed = engine.poison_training(untrusted, rate=0.60, seed=42)
        expected_n = int(n_spam * 0.60)  # floor
        assert len(changed) == expected_n
        assert len(set(changed)) == len(changed)
        before = untrusted.set_index("id")["label"]
        after = out.set_index("id")["label"]
        flipped = [i for i in changed]
        for cid in flipped:
            assert before.loc[cid] == "spam"
            assert after.loc[cid] == "ham"
        unchanged = set(out["id"]) - set(changed)
        for cid in unchanged:
            assert before.loc[cid] == after.loc[cid]

    def test_rate_zero_identical(self, prepared):
        untrusted = prepared["untrusted"]
        out, changed = engine.poison_training(untrusted, rate=0.0, seed=42)
        assert changed == []
        pd.testing.assert_frame_equal(
            out.reset_index(drop=True), untrusted.reset_index(drop=True))

    def test_rate_one_flips_all_spam(self, prepared):
        untrusted = prepared["untrusted"]
        n_spam = int((untrusted["label"] == "spam").sum())
        out, changed = engine.poison_training(untrusted, rate=1.0, seed=42)
        assert len(changed) == n_spam
        assert (out["label"] == "ham").all()

    def test_invalid_rates_rejected(self, prepared):
        untrusted = prepared["untrusted"]
        for bad in (-0.1, 1.1, 2.0):
            with pytest.raises(ValueError):
                engine.poison_training(untrusted, rate=bad, seed=42)

    def test_deterministic_for_same_seed(self, prepared):
        untrusted = prepared["untrusted"]
        _, c1 = engine.poison_training(untrusted, rate=0.60, seed=42)
        _, c2 = engine.poison_training(untrusted, rate=0.60, seed=42)
        assert c1 == c2
        _, c3 = engine.poison_training(untrusted, rate=0.60, seed=7)
        assert c1 != c3


class TestFilterSuspicious:
    def test_returns_three_objects(self, prepared):
        retained, quarantined, ref = engine.filter_suspicious(
            prepared["trusted"], prepared["untrusted"], threshold=0.65)
        assert isinstance(retained, pd.DataFrame)
        assert isinstance(quarantined, pd.DataFrame)
        from sklearn.pipeline import Pipeline
        assert isinstance(ref, Pipeline)

    def test_retained_and_quarantined_partition_untrusted(self, prepared):
        retained, quarantined, _ = engine.filter_suspicious(
            prepared["trusted"], prepared["untrusted"], threshold=0.65)
        assert set(retained["id"]) | set(quarantined["id"]) == set(prepared["untrusted"]["id"])
        assert set(retained["id"]) & set(quarantined["id"]) == set()
        assert list(retained.columns) == ["id", "label", "text"]

    def test_quarantined_has_reference_probability_column(self, prepared):
        _, quarantined, _ = engine.filter_suspicious(
            prepared["trusted"], prepared["untrusted"], threshold=0.65)
        assert "reference_spam_probability" in quarantined.columns
        if len(quarantined):
            probs = quarantined["reference_spam_probability"]
            assert (probs >= 0.0).all() and (probs <= 1.0).all()
            assert (probs >= 0.65).all()

    def test_only_observed_ham_labels_quarantined(self, prepared):
        _, quarantined, _ = engine.filter_suspicious(
            prepared["trusted"], prepared["untrusted"], threshold=0.65)
        # filter operates on the OBSERVED (possibly poisoned) labels only
        assert set(quarantined["label"].unique()) <= {"ham"}

    def test_defence_does_not_use_clean_labels_or_manifest(self, prepared):
        """Poison first, then filter on poisoned labels; the defence must
        quarantine some flipped rows without access to original labels."""
        poisoned, changed = engine.poison_training(
            prepared["untrusted"], rate=0.60, seed=42)
        retained, quarantined, _ = engine.filter_suspicious(
            prepared["trusted"], poisoned, threshold=0.65)
        # all quarantined rows currently labeled ham
        assert set(quarantined["label"].unique()) <= {"ham"}
        # defence is non-trivial: it catches at least some poisoned rows
        caught = set(quarantined["id"]) & set(changed)
        assert len(caught) > 0
        assert len(retained) + len(quarantined) == len(poisoned)

    def test_threshold_bounds_validated(self, prepared):
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError):
                engine.filter_suspicious(
                    prepared["trusted"], prepared["untrusted"], threshold=bad)

    def test_deterministic(self, prepared):
        r1, q1, _ = engine.filter_suspicious(
            prepared["trusted"], prepared["untrusted"], threshold=0.65)
        r2, q2, _ = engine.filter_suspicious(
            prepared["trusted"], prepared["untrusted"], threshold=0.65)
        assert list(q1["id"]) == list(q2["id"])
        assert list(r1["id"]) == list(r2["id"])
