import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/clean_resume_data.csv")

# -----------------------------
# Handle Missing Values
# -----------------------------
df["Feature"] = df["Feature"].fillna("").astype(str)

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    return " ".join(text)

df["cleaned"] = df["Feature"].apply(clean_text)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)

X = tfidf.fit_transform(df["cleaned"])
y = df["Category"]

print("TF-IDF Shape:", X.shape)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model Training (IMPROVED)
# -----------------------------
model = LinearSVC()

model.fit(X_train, y_train)
# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# Accuracy
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
#
joblib.dump(model, "models/resume_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("Model Saved Successfully!")