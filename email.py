# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Predict function
def predict_email(model, vectorizer, email_text):
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]  # 0 = Ham, 1 = Spam
    return prediction
# Streamlit web application
def main():
    st.title("Email Spam Detection")

    # Instructions for the app
    st.write("""
    Enter the email text below and click 'Predict' to check if it's spam or not.
    This app uses a machine learning model trained with various algorithms to detect spam.
    """)

    # Input box for email content
    email_input = st.text_area("Enter the email content here:")

    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # Predict button
    if st.button("Predict"):
        if email_input:
            # Perform spam prediction
            prediction = predict_email(model, vectorizer, email_input)
            if prediction == 1:
                st.error("This email is classified as SPAM.")
                st.image('images/spam.png', caption="Warning: SPAM detected!",width=200)
            else:
                st.success("This email is classified as NON-SPAM.")
                st.image('images/nonspam.jpg', caption="This is a valid email.",width=200)
        else:
            st.warning("Please enter some email content to classify.")

# Run the app
if __name__ == '__main__':
    main()