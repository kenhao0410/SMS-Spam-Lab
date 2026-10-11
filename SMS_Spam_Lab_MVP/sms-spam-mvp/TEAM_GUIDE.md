# Team handoff

This MVP gives us a common pipeline and a visible experiment. It is not the final submission. The example attack and defence are included so everyone can see the integration points.

## Agreed broad roles

Dipta builds the base SMS classifier. Two members work on attacks and two on defences. Names and the specific methods still need to be agreed.

Suggested division within those pairs, not a recorded group decision:

- Attack member A: reproduce label-flipping at several attack budgets and seeds. Describe the attacker's exact access and objective.
- Attack member B: develop a genuinely different threat, such as targeted label-flipping or a lab-only trigger/backdoor, with a separate success metric. Do not call random label noise a backdoor.
- Defence member A: extend the training-data checks and suspicious-label quarantine. Show which rows are excluded and why.
- Defence member B: compare another training or review strategy, include the cost and false-positive consequences, and test failure cases.

Each member owns their method, tests, report section and presentation. Pair work needs review by the other member, and Dipta checks integration. These suggestions are deliberately not extra finished components.

## Common data contract

Training DataFrame or CSV:

```text
id,label,text
<stable message id>,ham,<message>
<stable message id>,spam,<message>
```

The actual exported CSV contains quoted public corpus messages as needed. Do not build CSV by joining strings manually. `id` is a hash of the normalised SMS text; it is not a user identifier. For custom training through the CLI, `id` is optional.

Attack input is the untrusted TRAINING subset only. Attack output is a new training subset plus a manifest of changed IDs. Never overwrite the original corpus. Never add trusted or test rows to the attack pool.

Defence input is the protected reference TRAINING subset and the observed untrusted subset. Defence output is retained rows plus quarantined rows/reasons. It does not get the attack manifest, original labels of attacked rows, or test labels. The evaluator can inspect attack ground truth separately for analysis after a defence has been frozen.

## Engine entry points

- `prepare_data()` returns the frozen training/test and trusted/untrusted split.
- `fit_model(training_frame)` accepts `label,text` and returns the full sklearn pipeline.
- `poison_training(untrusted, rate, seed)` is the replaceable example attack.
- `filter_suspicious(trusted, untrusted, threshold)` is the replaceable example defence.
- `evaluate(model, test)` measures performance on the clean holdout.
- `run_experiment(prepared, poison_rate, defence_threshold, seed)` runs all three conditions.

Use `report_for_json(result, prepared)` for machine-readable evidence. Save parameters, random seed, corpus checksum and test fingerprint with each report. Do not load the test set from an attack or defence function.

## Why preprocessing needs a specific threat model

Removing punctuation or lowercasing text does not repair wrong training labels. The current example instead checks whether an observed label disagrees strongly with a separate model trained on a protected clean reference set. It removes suspicious rows before retraining rather than pretending to know their original labels.

Its protection depends on having verified reference labels. This is an assumption the report must state. A network IDS is not automatically a defence against malicious labels in an offline CSV.

## Next group decisions

Agree the final threat model, both attack methods, both defence methods, experiment budgets and reporting responsibilities. Add a defence-on-clean-data control, repeated seeds and training-only validation. Reserve a fresh final evaluation set before any tuning; this MVP's visible test set is exploratory once people start adjusting settings against it.

The report should explain why an attack succeeded or failed. A defence that does not improve recall, or that wrongly removes normal messages, is still a result to document honestly.
