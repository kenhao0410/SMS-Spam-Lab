# SMS Spam Lab

A local MVP for the SE6022 team. It shows a real SMS classifier and a small training-data poisoning experiment. It does not use an LLM, a paid API or a GPU.

This is a starting point for discussion, not the team's final attack/defence implementation or a production scam detector. All UI text is in English.

## Run the demo

Use Python 3.12. Internet is needed to install dependencies. The UCI dataset is included, so the app can run without internet afterwards.

### Windows PowerShell

Open a terminal inside the extracted `sms-spam-mvp` folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS or Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local address shown in the terminal, normally http://localhost:8501. Stop the app with Ctrl+C. The default configuration listens only on your own computer, not the public network. No account or API key is required.

## Show it to the team in five minutes

1. Open **Try a message**. Use a labelled held-out corpus example or type an English SMS. Explain that ham means a normal message. A model score is not a guarantee that a message is safe.
2. Open **Run an experiment**, leave the settings unchanged and click **Run experiment**. These are actual model fits, not a scripted animation.
3. Compare Baseline, Poisoned and Defended on the same clean test set. Focus on spam recall and missed spam, not accuracy alone.
4. Expand the quarantined rows. Explain that the defence can make mistakes too.
5. Open **Team handoff** and agree who extends each part.

## What the MVP does

- Parses the official UCI SMS Spam Collection as tab-separated label/message records.
- Groups case-folded, whitespace-normalised duplicate messages before splitting. Conflicting-label groups are excluded. This helps prevent identical messages leaking between training and test; it does not remove all near-duplicate campaigns.
- Uses one stratified 80/20 train/test split with seed 42. The test set is never poisoned, used for fitting the vocabulary or used to choose the defence threshold.
- Fits a TF-IDF word unigram/bigram vectoriser and Logistic Regression. Prediction threshold is the model's default 0.5.
- Reserves a protected, correctly labelled reference subset containing 25 percent of the training rows. This is a declared defence assumption, not hidden access to test labels.
- Simulates an attacker who can change labels in the remaining training pool. The attack flips a configurable fraction of **spam rows in that untrusted pool** to ham. It does not change the SMS text.
- Fits a separate reference model using only the protected subset. It quarantines untrusted rows labelled ham when that reference model assigns a spam score at least 0.65 by default. It retrains on the protected rows plus the remaining untrusted rows.
- Measures spam precision, recall and F1, accuracy, normal-message false positives, missed spam and confusion matrices. Filtering is not guaranteed to help.

Changing the slider selects a new experiment; it must be explicitly run. The page shows the settings attached to the displayed results. Repeatedly adjusting settings against this same test set is exploratory and must not be presented as an unbiased final evaluation.

## Command-line pipeline

Generate reproducible reports and the baseline model:

```bash
python cli.py experiment --poison-rate 0.6 --output outputs
python cli.py predict --text "Are we still meeting after class?"
```

Here `python` means the virtual-environment Python shown above. The experiment writes `report.json`, `metrics.csv`, `model.joblib`, `poisoned_training.csv`, `quarantine.csv`, `attack_manifest.json` and `test_manifest.json` under the selected output directory. `model.joblib` contains the complete baseline TF-IDF and classifier pipeline, not just classifier weights. The other two models can be retrained from the same experiment.

Train using another team member's training CSV:

```bash
python cli.py train --csv path/to/training.csv --output outputs/custom.joblib
python cli.py predict --model outputs/custom.joblib --text "Are we still meeting after class?"
```

CSV requires `label,text` columns; labels must be `ham` or `spam`, and both classes must remain. `id` is optional. Never add held-out test messages to training. Do not load a joblib file from an untrusted source: Python model files can execute code when loaded.

Run tests:

```bash
python -m pytest tests -q
```

## Files to read first

- `engine.py`: data preparation, training, attack, defence and metrics.
- `app.py`: the English walkthrough and interactive demo.
- `cli.py`: model/report export and custom training entry point.
- `TEAM_GUIDE.md`: interfaces and suggested work split.
- `data/SOURCE.md`: attribution, source URL and download checksum.
- `outputs/report.json`: a real recorded default run, not final research evidence.
- `QA.md`: verified runtime and browser checks for this delivery.

## Limits and privacy

This is an old, largely English spam/ham corpus. Its labels are not phishing labels. Good scores on this corpus do not prove detection of current scams, multilingual SMS, targeted evasion, clean-label poisoning or backdoors.

The illustrative defence needs a protected clean training subset. If that subset is wrong, too small or attacked, the assumption fails. It may remove legitimate messages or miss corrupted labels. The team's final study needs repeated seeds, attack budgets, a clean-data defence control, training-only validation and a separately reserved final evaluation set.

The local app does not intentionally save typed SMS messages or send them to an external service. Avoid pasting passwords, OTPs or private conversations into a shared screen. Only public corpus rows are included in exports. Streamlit usage statistics are disabled in the project config. There is no authentication; do not expose this MVP publicly without a separate hosting/security review.

## Dataset credit

Tiago Almeida and Jose Maria Gomez Hidalgo, SMS Spam Collection, UCI Machine Learning Repository, https://doi.org/10.24432/C5CC84. UCI lists the dataset under CC BY 4.0. The original corpus and readme are bundled unchanged; deduplication, splitting and poisoning happen in memory or explicitly labelled experiment outputs.
