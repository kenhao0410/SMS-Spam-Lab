from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine


@pytest.fixture(scope='module')
def data():
    return engine.prepare_data()


def test_real_experiment_preserves_partitions_and_has_all_conditions(data):
    before = {name: data[name].copy(deep=True) for name in ['train', 'test', 'trusted', 'untrusted']}
    assert hasattr(engine, 'run_experiment'), 'Experiment orchestration is missing'
    result = engine.run_experiment(data)
    assert set(result['models']) == {'Baseline', 'Poisoned', 'Defended'}
    assert set(result['metrics']) == set(result['models'])
    assert result['summary']['test_fingerprint'] == data['test_fingerprint']
    assert result['summary']['changed_rows'] == len(result['attack_manifest'])
    assert len(result['poisoned_train']) == len(data['train'])
    assert set(result['quarantined']['id']) <= set(data['untrusted']['id'])
    for name in before:
        pd.testing.assert_frame_equal(before[name], data[name])


def test_experiment_rejects_holdout_overlap(data):
    bad = dict(data)
    bad['untrusted'] = pd.concat([data['untrusted'], data['test'].iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match='overlap|partition'):
        engine.run_experiment(bad)


def test_cli_experiment_exports_real_files(tmp_path):
    assert (ROOT / 'cli.py').is_file(), 'The command-line pipeline is missing'
    run = subprocess.run([sys.executable, str(ROOT / 'cli.py'), 'experiment', '--output', str(tmp_path)],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    for name in ['report.json', 'metrics.csv', 'model.joblib', 'poisoned_training.csv',
                 'quarantine.csv', 'attack_manifest.json', 'test_manifest.json']:
        assert (tmp_path / name).is_file(), name
    report = json.loads((tmp_path / 'report.json').read_text())
    assert set(report['metrics']) == {'Baseline', 'Poisoned', 'Defended'}
    assert report['summary']['changed_rows'] > 0
    predicted = subprocess.run([sys.executable, str(ROOT / 'cli.py'), 'predict', '--model',
                                str(tmp_path / 'model.joblib'), '--text', 'Are we still meeting after class?'],
                               capture_output=True, text=True, timeout=60)
    assert predicted.returncode == 0, predicted.stderr
    assert json.loads(predicted.stdout)['label'] in {'ham', 'spam'}


def test_custom_train_roundtrip_reuses_pipeline(data, tmp_path):
    csv = tmp_path / 'training.csv'
    data['train'].to_csv(csv, index=False)
    model_path = tmp_path / 'custom.joblib'
    run = subprocess.run([sys.executable, str(ROOT / 'cli.py'), 'train', '--csv', str(csv),
                          '--output', str(model_path)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    import joblib
    model = joblib.load(model_path)
    assert 'tfidf' in model.named_steps and 'clf' in model.named_steps
    assert engine.evaluate(model, data['test']) == engine.evaluate(engine.fit_model(data['train']), data['test'])


def test_custom_train_rejects_test_message_even_without_id(data, tmp_path):
    csv = tmp_path / 'contaminated.csv'
    pd.concat([data['train'], data['test'].iloc[:1]], ignore_index=True)[['label', 'text']].to_csv(csv, index=False)
    run = subprocess.run([sys.executable, str(ROOT / 'cli.py'), 'train', '--csv', str(csv),
                          '--output', str(tmp_path / 'bad.joblib')], capture_output=True, text=True, timeout=60)
    assert run.returncode != 0
    assert 'holdout' in run.stderr
    assert not (tmp_path / 'bad.joblib').exists()


def test_zero_poison_is_a_clean_control(data):
    result = engine.run_experiment(data, poison_rate=0)
    assert result['summary']['changed_rows'] == 0
    assert result['metrics']['Baseline'] == result['metrics']['Poisoned']
    encoded = json.dumps(engine.report_for_json(result, data))
    assert json.loads(encoded)['data']['test_rows'] == len(data['test'])


def test_full_untrusted_spam_attack_still_runs(data):
    result = engine.run_experiment(data, poison_rate=1)
    assert result['summary']['changed_rows'] == int((data['untrusted']['label'] == 'spam').sum())
    assert len(result['models']) == 3
