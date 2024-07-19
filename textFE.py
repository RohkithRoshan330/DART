import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import string

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize text
    words = [word for word in words if word.isalpha()]  # Remove numbers and other non-alphabetic tokens
    return ' '.join(words)

# Function to compute N-grams and TF-IDF
def compute_tfidf(corpus, ngram_range=(1, 2), top_n=10):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)
    tfidf_scores = X.sum(axis=0).A1
    tfidf_scores_df = pd.DataFrame({
        'ngram': vectorizer.get_feature_names_out(),
        'tfidf': tfidf_scores
    })
    top_ngrams = tfidf_scores_df.sort_values(by='tfidf', ascending=False).head(top_n)
    return top_ngrams

# Streamlit app
st.title("N-grams and TF-IDF Feature Engineering")

# File uploader for text or CSV files
uploaded_file = st.file_uploader("Upload a text file or CSV file containing text data", type=["txt", "csv"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        # Process text file
        text_data = uploaded_file.read().decode("utf-8")
        corpus = [preprocess_text(text_data)]
    elif uploaded_file.type == "text/csv":
        # Process CSV file
        df = pd.read_csv(uploaded_file)
        text_data = ' '.join(df.astype(str).apply(' '.join, axis=1))
        corpus = [preprocess_text(text_data)]

    # Input for N-gram range and top N results
    ngram_min = st.number_input("Min N-gram", min_value=1, max_value=5, value=1)
    ngram_max = st.number_input("Max N-gram", min_value=1, max_value=5, value=2)
    top_n = st.number_input("Top N results", min_value=1, max_value=50, value=10)

    if st.button("Compute TF-IDF"):
        with st.spinner("Computing TF-IDF..."):
            # Compute TF-IDF for the corpus
            top_ngrams = compute_tfidf(corpus, ngram_range=(ngram_min, ngram_max), top_n=top_n)
            st.success("TF-IDF computation complete!")

            # Display top N-grams with their TF-IDF scores
            st.write(f"Top {top_n} N-grams by TF-IDF score:")
            st.dataframe(top_ngrams)

            # Show a downloadable CSV with the results
            csv = top_ngrams.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='top_ngrams_tfidf.csv',
                mime='text/csv',
            )
