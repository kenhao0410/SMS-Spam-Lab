# Verified MVP checks

Verification environment: Linux ARM64, Python 3.12.3. This is a local demonstration, not a production deployment. Windows, macOS and physical Safari have not been exercised here.

## Runtime evidence

- `python -m pytest tests -q`: **55 passed**. Includes real-corpus cleaning/splits, actual training, vocabulary fitting, poisoning copy integrity, defence partitions, CLI exports, model roundtrip, holdout-overlap rejection and Streamlit state/error flows.
- `python cli.py experiment --output outputs`: completed on the bundled original UCI corpus.
- `python cli.py predict --text "Are we still meeting after class?"`: ham, model spam score 0.020546168282841944.
- Independent manual composition of attack, filter and training reproduced all metrics in the exported report. Input train/test/trusted/untrusted frames remained unchanged; attack and quarantine IDs are confined to the untrusted training pool.

The original file checksum is `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`. It contains 5574 records. Cleaning retains 5159 normalised unique messages, removes 415 duplicates, and finds no malformed rows or conflicting-label groups. The fixed split has 4127 training rows and 1032 test rows; 1031 training rows are the protected reference set and 3096 are untrusted.

## Real default experiment

Seed 42, 60 percent of untrusted spam labels changed to ham, quarantine threshold 0.65. This changes 231 labels and quarantines 193 rows. It is about 5.597 percent of all training rows, not 60 percent of all training rows.

- Baseline: spam precision 0.9732142857142857, recall 0.8515625, F1 0.9083333333333333; 19 missed spam and 3 normal messages flagged.
- Poisoned: precision 1.0, recall 0.328125, F1 0.49411764705882355; 86 missed spam and 0 normal messages flagged.
- Defended: precision 1.0, recall 0.765625, F1 0.8672566371681416; 30 missed spam and 0 normal messages flagged.

All conditions use the same clean test fingerprint: `0c6227d3c6b27d24bc6571084ed63311bf004970935c46be07903657056a093b`.

These are one-run exploratory results. The protected-reference assumption is strong; this is not proof of resistance to arbitrary poisoning, backdoors, modern scams or phishing.

## Browser evidence

Headless Chromium was exercised at 1440x1000, 768x1024, 375x812 and 812x375. All three views had zero page-level horizontal overflow and zero page errors at those sizes.

Exercised: blank input, custom prediction, real held-out sample selection, keyboard activation, actual experiment runs, downloaded JSON matching independently computed metrics, quarantine inspection, confusion matrices, model-specific mistakes and team handoff. At desktop size, changing the attack slider left the saved-run label unchanged until submission; rerunning updated it to 10 percent, and changing the quarantine threshold updated the saved run to 0.70.

A separate browser run blocked any external HTTP requests. Prediction and model training still worked; no external requests, failed requests or console errors occurred. Out-of-vocabulary input showed its warning.

The main action measures 44px high. The textarea uses 16px text and a visible boundary. Keyboard focus is a 3px deep-teal outline. At 200 percent root text size on a 375px viewport, paragraph text measured 32px and document width remained 375px.

Contrast checks: main text on paper 14.35:1; white on teal 6.91:1; teal on paper 6.68:1; main text on the secondary surface 13.09:1. The input boundary has 4.19:1 against the secondary surface, above the non-text minimum of 3:1. The border colour is not used for body text.

Parent visual review found the comparison, settings and defence assumption readable, with no visible overlap or clipping. The screenshot is appearance evidence only; interaction evidence comes from the browser tests.

## Delivery check

The delivery ZIP uses an explicit file allowlist. Virtual environments, caches, credentials, unrelated coursework and QA scratch files are excluded. Original source attribution is included. Python model files must be treated as trusted-code artifacts, not safe arbitrary uploads.

Extracted-package execution: PASS. The archive was extracted into an independent folder; all 55 tests passed there. CLI experiment generation and prediction using the newly saved model also passed. The imported engine path was verified to point to the extracted copy, not the original workspace. The final repack changes only this verification note; executable sources and bundled data remain identical to the exercised archive.

## Anti-Slop delivery gate

- Hard gate: PASS. Sourced corpus, measured metrics, real controls, clear loading/error/empty states and explicit limitations; browser overflow and error checks passed.
- Purpose gate: PASS. DESIGN.md records the lab-notebook direction, readable hierarchy, restrained palette and why there are no decorative or marketing sections.
- Liveliness: PASS. ENERGY 1 / RHYTHM 2 / MOTION 1; message form and experiment comparison are the focal points, with a deliberate teal action colour.
- Craftsmanship: PASS. Real model fits and downloads; desktop/mobile/keyboard/zoom checks; no invented performance or safety guarantees. Physical-device and cross-platform claims are explicitly withheld.
