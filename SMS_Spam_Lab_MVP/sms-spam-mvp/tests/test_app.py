from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_opens_with_clear_title():
    assert (ROOT / 'app.py').is_file(), 'The SMS demo app has not been built yet'
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert not app.exception
    assert any(title.value == 'SMS Spam Lab' for title in app.title)


def test_message_form_rejects_blank_input():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert any(area.label == 'SMS message' for area in app.text_area), 'Message form is missing'
    app.text_area[0].set_value('   ')
    app.button(key='classify').click().run(timeout=60)
    assert any('Enter a message' in warning.value for warning in app.warning)


def test_message_form_predicts_without_error():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    app.text_area[0].set_value('Are we still meeting after class?')
    app.button(key='classify').click().run(timeout=60)
    assert not app.exception
    assert any('Prediction:' in item.value for item in app.markdown)


def test_experiment_has_real_run_control():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert not app.exception
    app.radio(key='view').set_value('Run an experiment').run(timeout=60)
    assert any(button.label == 'Run experiment' for button in app.button), 'Experiment run control is missing'
    app.button(key='run_experiment').click().run(timeout=120)
    assert not app.exception
    assert any('Displayed run:' in caption.value for caption in app.caption)
    assert len(app.metric) == 3


def test_team_handoff_has_the_agreed_roles():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    app.radio(key='view').set_value('Team handoff').run(timeout=60)
    assert not app.exception
    assert any('Dipta builds the base SMS classifier.' in item.value for item in app.markdown)


def test_selected_corpus_example_really_fills_message():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    app.selectbox(key='example').set_value('Corpus spam example').run(timeout=60)
    assert app.text_area[0].value.strip()
    app.button(key='classify').click().run(timeout=60)
    assert not app.exception
    assert any('Prediction:' in item.value for item in app.markdown)


def test_experiment_keeps_saved_settings_until_explicit_rerun():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    app.radio(key='view').set_value('Run an experiment').run(timeout=60)
    app.button(key='run_experiment').click().run(timeout=120)
    assert not app.exception
    assert app.session_state['experiment']['summary']['poison_rate'] == 0.6
    app.slider(key='poison_rate').set_value(10)
    assert app.session_state['experiment']['summary']['poison_rate'] == 0.6
    assert any('Displayed run: 60%' in caption.value for caption in app.caption)
    app.button(key='run_experiment').click().run(timeout=120)
    assert not app.exception
    assert app.session_state['experiment']['summary']['poison_rate'] == 0.1


def test_can_switch_mistake_model_after_a_rerun():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    app.radio(key='view').set_value('Run an experiment').run(timeout=60)
    app.button(key='run_experiment').click().run(timeout=120)
    app.number_input(key='defence_threshold').set_value(0.70)
    app.button(key='run_experiment').click().run(timeout=120)
    assert any(radio.key == 'inspect_model' for radio in app.radio), 'A stable model selector is missing'
    app.radio(key='inspect_model').set_value('Poisoned').run(timeout=60)
    assert not app.exception
    assert any('Poisoned mistakes:' in item.value for item in app.markdown)


def test_data_error_is_explained_without_a_traceback(monkeypatch):
    import engine
    import streamlit as st
    st.cache_resource.clear()

    def invalid_corpus():
        raise ValueError('test-only invalid corpus fixture')

    monkeypatch.setattr(engine, 'prepare_data', invalid_corpus)
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert not app.exception
    assert any('Could not prepare the model' in error.value for error in app.error)
    assert any('requirements.txt' in item.value for item in app.info)
    st.cache_resource.clear()
