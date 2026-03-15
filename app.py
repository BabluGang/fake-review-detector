import streamlit as st
import pickle
import re
import os
import nltk

# download nltk data only if not already present
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(f'{nltk_data_path}/corpora/wordnet'):
    nltk.download('wordnet', quiet=True)
if not os.path.exists(f'{nltk_data_path}/corpora/stopwords'):
    nltk.download('stopwords', quiet=True)
if not os.path.exists(f'{nltk_data_path}/corpora/omw-1.4'):
    nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# load model and vectorizer
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# same cleaning function from training
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ---- UI ----
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍")

st.title("🔍 Fake Review Detector")
st.markdown("Paste a product review below to check if it's **genuine or fake**.")

review = st.text_area("📝 Enter Review", height=200, placeholder="e.g. This product is amazing! Best purchase I've ever made...")

if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # normalize decision function score to confidence %
        score = model.decision_function(vectorized)[0]
        from scipy.special import expit  # sigmoid function
        confidence = round(expit(abs(score)) * 100, 1)

        if prediction == 1:
            st.error(f"🚨 Fake Review detected! — Confidence: {confidence}%")
        else:
            st.success(f"✅ Genuine Review — Confidence: {confidence}%")

        # extra details
        with st.expander("See cleaned input"):
            st.write(cleaned)

st.markdown("---")
st.caption("Built with Linear SVM + TF-IDF | Trained on Amazon Reviews Dataset")