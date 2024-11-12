# ticket_classification_app.py
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk

# Download NLTK resources
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize necessary components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
cv = CountVectorizer()
minmax = MinMaxScaler()
le = LabelEncoder()
rf_clf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')


rf_clf = joblib.load('clf.pkl')
cv = joblib.load('cv.pkl')
minmax = joblib.load('minmax.pkl')
le = joblib.load('le.pkl')

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)

# Streamlit UI
st.title("Customer Support Ticket Classification")
st.write("This app classifies support tickets into predefined categories based on their content.")

# Input field for ticket text
ticket_text = st.text_area("Enter ticket text here:")

# Button to perform inference
if st.button("Classify Ticket"):
    if ticket_text:
        # Preprocess and transform input text
        preprocessed_text = preprocess_text(ticket_text)
        vectorized_text = cv.transform([preprocessed_text]).toarray()
        scaled_text = minmax.transform(vectorized_text)

        # Predict and display the output
        prediction = rf_clf.predict(scaled_text)
        predicted_class = le.inverse_transform(prediction)[0]

        st.write(f"Predicted Class: **{predicted_class}**")
    else:
        st.write("Please enter some text for classification.")
