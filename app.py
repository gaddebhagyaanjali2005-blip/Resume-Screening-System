import streamlit as st
import joblib
import re

# Load model
model = joblib.load("models/resume_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

st.title("Resume Screening System")

resume_text = st.text_area("Paste Resume Text Here")

if st.button("Predict"):
    cleaned = clean_text(resume_text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    st.success(f"Predicted Category: {prediction[0]}")