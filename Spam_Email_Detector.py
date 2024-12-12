import streamlit as st
import joblib
import re
import string

# Load the saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("Spam Email Detector")

# Input box for user to enter email text
email_text = st.text_area("Enter email text:")

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Prediction button
if st.button("Predict"):
    if email_text.strip():
        # Preprocess and vectorize the input
        email_text_cleaned = clean_text(email_text)
        vectorized_text = vectorizer.transform([email_text_cleaned])
        
        # Make prediction
        prediction = model.predict(vectorized_text)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        st.write(f"Prediction: **{result}**")
    else:
        st.write("Please enter text for prediction.")
