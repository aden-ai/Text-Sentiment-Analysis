import re
import nltk
nltk.download('wordnet') # Download the required 'wordnet' resource
from nltk.stem import WordNetLemmatizer

def clean_text(text):
  # Remove HTML tags
  text = re.sub(r'<.*?>', '', text)
  # Remove punctuation and special characters
  text = re.sub(r'[^\w\s]', '', text)
  # Lemmatize to root form
  lemmatizer = WordNetLemmatizer()
  text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
  return text.lower().strip()