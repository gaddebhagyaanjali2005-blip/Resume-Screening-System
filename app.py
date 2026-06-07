import streamlit as st
import joblib
import re

# Load saved model and vectorizer
model = joblib.load("models/resume_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

# Streamlit UI
st.title("📄 Resume Screening System")

st.write("Paste resume text below and click Predict.")

resume_text = st.text_area("Resume Text")

if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:
        cleaned = clean_text(resume_text)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)

        st.success(f"Predicted Category: {prediction[0]}")