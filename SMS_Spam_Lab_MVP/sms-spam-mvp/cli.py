from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import joblib
import pandas as pd

from engine import (canonical_text, fit_model, predict_sms, prepare_data,
                    report_for_json, run_experiment)

ROOT = Path(__file__).resolve().parent


def experiment(args):
    prepared = prepare_data()
    result = run_experiment(prepared, poison_rate=args.poison_rate,
                            defence_threshold=args.defence_threshold, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = report_for_json(result, prepared)
    report['environment']['python'] = platform.python_version()
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = [{'model': name, **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}}
            for name, metrics in result['metrics'].items()]
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    joblib.dump(result['models']['Baseline'], output / 'model.joblib')
    result['poisoned_train'].to_csv(output / 'poisoned_training.csv', index=False)
    result['quarantined'].to_csv(output / 'quarantine.csv', index=False)
    (output / 'attack_manifest.json').write_text(json.dumps(result['attack_manifest'], indent=2))
    (output / 'test_manifest.json').write_text(json.dumps({
        'data_split_seed': prepared['seed'], 'fingerprint': prepared['test_fingerprint'],
        'ids': prepared['test']['id'].tolist()}, indent=2))
    print(json.dumps({'output': str(output.resolve()), 'metrics': result['metrics'],
                      'summary': result['summary']}, indent=2))


def training_frame(csv_path):
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if not {'label', 'text'} <= set(frame.columns):
        raise ValueError('Training CSV needs label,text columns')
    if not frame['text'].map(lambda value: bool(value.strip())).all():
        raise ValueError('Training CSV contains empty text')
    if not set(frame['label']) <= {'ham', 'spam'}:
        raise ValueError('Training labels must be exactly ham or spam')
    frame['_key'] = frame['text'].map(canonical_text)
    if (frame.groupby('_key')['label'].nunique() > 1).any():
        raise ValueError('Training CSV has conflicting labels for the same message')
    expected_ids = frame['_key'].map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    if 'id' in frame and not (frame['id'] == expected_ids).all():
        raise ValueError('Provided IDs must match the canonical text SHA-256; omit id if unavailable')
    frame['id'] = expected_ids
    holdout_keys = set(prepare_data()['test']['text'].map(canonical_text))
    if set(frame['_key']) & holdout_keys:
        raise ValueError('Training CSV overlaps the fixed clean holdout; remove test messages')
    return frame.drop_duplicates('_key')[['id', 'label', 'text']]


def train(args):
    frame = training_frame(args.csv)
    model = fit_model(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(json.dumps({'model': str(output.resolve()), 'training_rows': len(frame)}, indent=2))


def predict(args):
    model_path = Path(args.model)
    if not model_path.is_file():
        raise ValueError(f'Model not found: {model_path}. Run the experiment command first.')
    # Only load model files produced by you or trusted team members.
    model = joblib.load(model_path)
    print(json.dumps(predict_sms(model, args.text), indent=2))


def main():
    parser = argparse.ArgumentParser(description='Local SMS spam training and poisoning lab')
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('experiment', help='Train and compare all three conditions')
    command.add_argument('--poison-rate', type=float, default=0.6)
    command.add_argument('--defence-threshold', type=float, default=0.65)
    command.add_argument('--seed', type=int, default=42, help='Attack sampling seed; data split remains 42')
    command.add_argument('--output', default=str(ROOT / 'outputs'))
    command.set_defaults(action=experiment)
    command = commands.add_parser('train', help='Train on a team member training CSV')
    command.add_argument('--csv', required=True)
    command.add_argument('--output', default=str(ROOT / 'outputs/custom.joblib'))
    command.set_defaults(action=train)
    command = commands.add_parser('predict', help='Use a trusted saved pipeline')
    command.add_argument('--model', default=str(ROOT / 'outputs/model.joblib'))
    command.add_argument('--text', required=True)
    command.set_defaults(action=predict)
    args = parser.parse_args()
    try:
        args.action(args)
    except (OSError, ValueError, pd.errors.ParserError, EOFError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
