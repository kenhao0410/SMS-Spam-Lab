from pathlib import Path

import streamlit as st

from engine import fit_model, predict_sms, prepare_data

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='SMS Spam Lab', layout='centered')
st.markdown('''<style>
button {min-height: 44px !important; min-width: 44px !important;}
textarea, input[role="combobox"], input[type="number"] {font-size: 1rem !important;}
input[type="range"], label:has(input[type="radio"]) {min-height: 44px !important;}
[data-testid="stTextArea"] div:has(> textarea),
[data-testid="stNumberInput"] div:has(> input[type="number"]) {
    border-color: #687671 !important;
}
div:has(> input[role="combobox"]) {
    border: 1px solid #687671 !important; border-radius: 4px;
    min-height: 44px !important;
}
button:focus-visible, textarea:focus-visible, input:focus-visible {
    outline: 3px solid #0F655D !important; outline-offset: 3px;
}
[data-testid="stMainBlockContainer"] {max-width: 64rem; padding-top: 2.5rem;}
[data-testid="stMarkdownContainer"] p {line-height: 1.6;}
[data-testid="stRadio"] label {min-height: 44px; align-items: center;}
@media (max-width: 600px) {
    [data-testid="stMainBlockContainer"] {padding: 1.25rem 1rem 3rem;}
    h1 {font-size: 2rem;}
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {animation: none !important; transition: none !important;}
}
</style>''', unsafe_allow_html=True)
st.title('SMS Spam Lab')
st.write('Train a spam classifier, change its training labels, and compare what happens.')
st.caption('SE6022 team MVP. Real UCI corpus, local models, no LLM or paid API.')


@st.cache_resource(show_spinner='Preparing the fixed data split and training the baseline...')
def baseline_state():
    prepared = prepare_data()
    return prepared, fit_model(prepared['train'])


def load_example(prepared):
    choice = st.session_state.example
    if choice != 'Write your own':
        label = 'ham' if choice == 'Corpus ham example' else 'spam'
        st.session_state.sms_text = prepared['test'].loc[
            prepared['test']['label'] == label, 'text'
        ].iloc[0]
    st.session_state.pop('prediction', None)


try:
    prepared, baseline = baseline_state()
except (OSError, ValueError) as error:
    st.error(f'Could not prepare the model: {error}')
    st.info('Keep data/SMSSpamCollection in this folder and install requirements.txt, then reload.')
    st.stop()

view = st.radio('Choose a view', ['Try a message', 'Run an experiment', 'Team handoff'], key='view')
st.divider()

if view == 'Try a message':
    st.header('Is this SMS spam?')
    st.write('Ham means a normal message. Try a corpus example or type an English SMS.')
    st.selectbox('Load a held-out example', ['Write your own', 'Corpus ham example', 'Corpus spam example'],
                 key='example', on_change=load_example, args=(prepared,))
    with st.form('message_form'):
        text = st.text_area('SMS message', key='sms_text', height=140, max_chars=5000,
                            placeholder='Type a message to classify. Do not paste passwords or OTPs.')
        submitted = st.form_submit_button('Classify message', key='classify', type='primary')
    if submitted:
        try:
            st.session_state.prediction = predict_sms(baseline, text)
        except ValueError as error:
            st.session_state.pop('prediction', None)
            st.warning('Enter a message before classifying it.' if not text.strip() else str(error))
    if result := st.session_state.get('prediction'):
        st.markdown(f"**Prediction: {result['label'].upper()}**")
        st.metric('Model spam score', f"{result['spam_probability']:.1%}")
        st.caption('Baseline model. This score is not calibrated proof of safety or fraud.')
        if result.get('warning'):
            st.warning(result['warning'])
    with st.expander('Data and limitations'):
        stats = prepared['stats']
        st.write(f"{stats['raw_rows']:,} original messages; {stats['unique_rows']:,} retained after cleaning. "
                 f"{stats['train_rows']:,} training rows and {stats['test_rows']:,} held-out test rows.")
        st.write('Case-folded, whitespace-normalised duplicates are grouped before splitting. '
                 'TF-IDF is fitted on training data only. Near-duplicate campaigns may remain.')
        st.write('This older, largely English dataset labels spam and ham, not phishing. '
                 'It is not evidence that the model detects modern scams or multilingual SMS.')
        st.markdown('[UCI dataset and attribution](https://doi.org/10.24432/C5CC84)')
        st.caption(f"Corpus SHA-256: {stats['corpus_sha256']}")
        st.caption(f"Test fingerprint: {prepared['test_fingerprint']}")
elif view == 'Run an experiment':
    st.header('What changes after poisoning?')
    st.write('The attacker relabels spam as ham in a copy of the untrusted training pool. '
             'The three models are evaluated on the same clean test set.')
    st.info('Defence assumption: 25 percent of the training set is protected and correctly labelled. '
            'A reference model trained only on those rows flags suspicious ham labels. '
            'Neither the reference model nor retraining uses test rows or the attack manifest.')
    with st.form('experiment_form'):
        rate = st.slider('Spam labels flipped in untrusted training pool (%)', 0, 100, 60, 10, key='poison_rate')
        threshold = st.number_input('Reference model quarantine threshold', min_value=0.50,
                                    max_value=0.95, value=0.65, step=0.05, key='defence_threshold')
        st.caption('The attack percentage is a fraction of untrusted spam, not of all training rows. '
                   'Threshold 0.65 is an MVP preset, not selected using this test set.')
        run = st.form_submit_button('Run experiment', key='run_experiment', type='primary')
    if run:
        from engine import run_experiment
        try:
            with st.spinner('Training the clean, poisoned and defended models...'):
                st.session_state.experiment = run_experiment(
                    prepared, poison_rate=rate / 100, defence_threshold=threshold, seed=42)
            st.session_state.pop('prediction', None)
        except (ValueError, OSError) as error:
            st.error(f'Experiment failed: {error}. Check that both classes remain in training.')
    if result := st.session_state.get('experiment'):
        import json
        from engine import report_for_json
        summary = result['summary']
        st.caption(f"Displayed run: {summary['poison_rate']:.0%} of untrusted spam labels flipped; "
                   f"quarantine threshold {summary['defence_threshold']:.2f}; seed {summary['seed']}. "
                   'These are the saved run settings, not unsubmitted changes in the form.')
        st.write(f"Changed training labels: {summary['changed_rows']}. "
                 f"Quarantined training rows: {summary['quarantined_rows']}. "
                 f"Clean test rows: {prepared['stats']['test_rows']}.")
        for column, name in zip(st.columns(3), ['Baseline', 'Poisoned', 'Defended']):
            with column:
                metrics = result['metrics'][name]
                st.subheader(name)
                st.metric('Spam recall', f"{metrics['spam_recall']:.1%}")
                st.write(f"Precision: {metrics['spam_precision']:.1%}")
                st.write(f"Spam F1: {metrics['spam_f1']:.3f}")
                st.write(f"Missed spam: {metrics['false_negatives']}")
                st.write(f"Normal messages flagged: {metrics['false_positives']}")
        st.warning('One exploratory run is not proof of robustness. The defence can miss poisoned '
                   'labels or quarantine genuine ham. Repeated tuning against this visible test set '
                   'would invalidate it as an unbiased final evaluation.')
        with st.expander('Confusion matrices and accuracy'):
            for name, metrics in result['metrics'].items():
                st.write(f"{name}: accuracy {metrics['accuracy']:.1%}")
                tn, fp = metrics['confusion_matrix'][0]
                fn, tp = metrics['confusion_matrix'][1]
                st.write(f'Actual ham: predicted ham {tn}, predicted spam {fp}. '
                         f'Actual spam: predicted ham {fn}, predicted spam {tp}.')
        with st.expander('Inspect quarantined training rows'):
            quarantined = result['quarantined']
            st.write(f'Quarantined rows: {len(quarantined)}')
            st.caption('These rows are suspicious, not confirmed malicious. '
                       'reference_spam_probability is the protected reference model score. First 20 rows shown.')
            if quarantined.empty:
                st.info('No rows met this threshold. That does not mean the training data is clean.')
            else:
                st.dataframe(quarantined[['label', 'text', 'reference_spam_probability']].head(20),
                             hide_index=True, width='stretch')
        with st.expander('Inspect held-out mistakes'):
            inspect_model = st.radio('Model to inspect', ['Baseline', 'Poisoned', 'Defended'], key='inspect_model')
            test = prepared['test'].copy()
            test['prediction'] = result['models'][inspect_model].predict(test['text'])
            mistakes = test.loc[test['label'] != test['prediction'], ['label', 'prediction', 'text']]
            st.write(f'{inspect_model} mistakes: {len(mistakes)}. First 20 shown.')
            if mistakes.empty:
                st.info('No mistakes on this particular test split. This is not a guarantee on new messages.')
            else:
                st.dataframe(mistakes.head(20), hide_index=True, width='stretch')
        st.download_button('Download report JSON', json.dumps(report_for_json(result, prepared), indent=2),
                           file_name='sms-spam-experiment.json', mime='application/json', key='download_report')
    else:
        st.info('No experiment yet. Choose the settings above and click Run experiment.')
else:
    st.markdown((ROOT / 'TEAM_GUIDE.md').read_text())
