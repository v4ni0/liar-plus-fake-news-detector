import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter

import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter

st.set_page_config(page_title="Fake News Detector", layout="wide")

def stemmed_tokenizer(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def get_job_length(job_title):
    if pd.isna(job_title):
        return 0
    return len(str(job_title))

def get_speaker_length(speaker):
    if pd.isna(speaker):
        return 0
    return len(str(speaker))

def get_total_history(pants_on_fire, false, barely_true, half_true, mostly_true):
    return pants_on_fire + false + barely_true + half_true + mostly_true

weights = {
    'pants_on_fire': 0,
    'false': 0.2,
    'barely_true': 0.4,
    'half_true': 0.6,
    'mostly_true': 0.8
}

def get_truth_index(pants_on_fire, false,barely_true, half_true, mostly_true):
    total = get_total_history(pants_on_fire, false,  barely_true, half_true, mostly_true)
    if total == 0:
        return 0
    score = (
        (pants_on_fire * weights['pants_on_fire']) +
        (false * weights['false']) +
        (barely_true * weights['barely_true']) +
        (half_true * weights['half_true']) +
        (mostly_true * weights['mostly_true'])
    ) / total
    return score

def categorize_context(text):
    text = str(text).lower()
    if any(word in text for word in ['facebook', 'twitter', 'tweet', 'post', 'social media']):
        return 'social_media'
    if any(word in text for word in ['tv', 'interview', 'cnn', 'abc', 'nbc', 'fox', 'radio']):
        return 'broadcast'
    if any(word in text for word in ['news release', 'press', 'statement', 'newspaper']):
        return 'official'
    if any(word in text for word in ['mailer', 'flyer', 'ad', 'advertisement', 'commercial', 'campaign']):
        return 'campaign_ads'
    if any(word in text for word in ['speech', 'floor speech', 'debate', 'rally', 'conference']):
        return 'live'
    return 'other'

top6_states = ['florida', 'illinois', 'new york', 'ohio', 'texas', 'wisconsin']

def categorize_state_info(x):
    if pd.isna(x) or str(x).strip() == "":
        return 'other'
    target = str(x).strip().lower()
    for state in top6_states:
        if state in target:
            return state
    return 'other'

def count_pos_tags(text):
    text = str(text)
    tokens = word_tokenize(text)
    if not tokens:
        return {'proper_nouns': 0, 'adjectives': 0, 'other': 0}

    pos_tags = nltk.pos_tag(tokens)
    counts = Counter(tag for _, tag in pos_tags)
    total_tokens = len(tokens)

    pos_counts = {
        'proper_nouns': (counts.get('NNP', 0) + counts.get('NNPS', 0)) / total_tokens,
        'adjectives': sum(count for tag, count in counts.items() if tag.startswith('JJ')) / total_tokens,
        'other': 0
    }
    pos_counts['other'] = 1 - pos_counts['proper_nouns'] - pos_counts['adjectives']
    return pos_counts

def calculate_lexical_diversity(text):
    text = str(text).lower()
    if not text:
        return 0
    words = word_tokenize(text)
    if not words:
        return 0
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return len(set(lemmatized_words)) / len(lemmatized_words)

try:
    pipeline = joblib.load('xgboost_pipeline.joblib')
except FileNotFoundError:
    st.error("Model file not found")
    st.stop()

LABEL_MAP = {
    0: 'pants-fire', 1: 'false', 2: 'barely-true', 3: 'half-true', 4: 'mostly-true', 5: 'true'
}
CLASS_NAMES_ORDERED = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]

st.title("🔎 Fake News Detector")
st.markdown("Enter the details of a statement")

st.header("Enter Statement Details")

col1, col2 = st.columns(2)

with col1:
    statement = st.text_area("Statement", height=100, placeholder="this statement is true")
    speaker = st.text_input("Speaker", placeholder="barack-obama")
    job_title = st.text_input("Speaker's Job Title", placeholder="president")
    state_info = st.text_input("State Info", placeholder="Illinois")
    party_affiliation = st.selectbox("Party Affiliation", ['republican', 'democrat', 'none', 'other'])
    context = st.text_input("Context", placeholder="random context")

with col2:
    st.subheader("Speaker`s Credit History")
    mostly_true_count = st.number_input("Mostly-True Count",
                                        min_value=0,
                                        value=0)
    half_true_count = st.number_input("Half-True Count", min_value=0, value=0)
    barely_true_count = st.number_input("Barely-True Count",
                                        min_value=0,
                                        value=0)
    false_count = st.number_input("False Count", min_value=0, value=0)
    pants_on_fire_count = st.number_input("Pants-on-Fire Count",
                                          min_value=0,
                                          value=0)

if st.button("Predict Truthfulness", type="primary"):
    if not statement:
        st.warning("Please enter a none-empty statement")
    else:
        pos_features = count_pos_tags(statement)

        input_data = {
            'statement':
            statement,
            'party_affiliation':
            party_affiliation,
            'barely_true':
            barely_true_count,
            'false':
            false_count,
            'half_true':
            half_true_count,
            'mostly_true':
            mostly_true_count,
            'speaker_length':
            get_speaker_length(speaker),
            'job_title_length':
            get_job_length(job_title),
            'state_category':
            categorize_state_info(state_info),
            'total_history':
            get_total_history(pants_on_fire_count, false_count, barely_true_count,
                              half_true_count, mostly_true_count),
            'truth_index':
            get_truth_index(pants_on_fire_count, false_count, barely_true_count,
                            half_true_count, mostly_true_count),
            'context_group':
            categorize_context(context),
            'statement_length':
            len(statement),
            'proper_nouns':
            pos_features['proper_nouns'],
            'adjectives':
            pos_features['adjectives'],
            'other':
            pos_features['other'],
            'lexical_diversity':
            calculate_lexical_diversity(statement)
        }

        input_df = pd.DataFrame([input_data])

        prediction_index = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)

        predicted_class = CLASS_NAMES_ORDERED[prediction_index]

        st.subheader("Prediction Result")
        st.success(f"The statement is most likely: **{predicted_class}**")

        st.subheader("Prediction Probabilities")
        probability = pd.DataFrame(prediction_proba,
                                   columns=CLASS_NAMES_ORDERED).T
        probability.columns = ['Probability']
        st.bar_chart(probability)
