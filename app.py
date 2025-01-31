import streamlit as st
import joblib
from preprocess import clean_text

# Load the saved Logistic Regression model
model = joblib.load('logistic_senti.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to predict sentiment
def predict_sentiment(text):
    cleaned_text = clean_text(text)  # Implement your text cleaning function
    features = tfidf_vectorizer.transform([cleaned_text])  # Use your TF-IDF vectorizer
    prediction = model.predict(features)[0]
    return prediction

# Streamlit UI code
st.title('Text Sentiment Analysis')
user_input = st.text_input('Enter Your Text: ')

if st.button('Predict'):
    sentiment = predict_sentiment(user_input)
    if sentiment == 1:
        st.success('Positive sentiment!')
    else:
        st.error('Negative sentiment!')
