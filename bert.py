import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Function to compute Bag of Words
def compute_bag_of_words(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df

# Streamlit app
st.title("Bag of Words Feature Engineering")

# File uploader for text or CSV files
uploaded_file = st.file_uploader("Upload a text file or CSV file containing text data", type=["txt", "csv"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        # Process text file
        text_data = uploaded_file.read().decode("utf-8")
        sentences = [preprocess_text(line) for line in text_data.split('\n') if line.strip()]
    elif uploaded_file.type == "text/csv":
        # Process CSV file
        df = pd.read_csv(uploaded_file)
        text_data = df.astype(str).apply(' '.join, axis=1)
        sentences = [preprocess_text(text) for text in text_data]

    if st.button("Compute Bag of Words"):
        with st.spinner("Computing Bag of Words..."):
            # Compute Bag of Words
            bow_df = compute_bag_of_words(sentences)
            st.success("Bag of Words computation complete!")

            # Display Bag of Words
            st.write("Bag of Words:")
            st.dataframe(bow_df)

            # Show a downloadable CSV with the Bag of Words
            csv = bow_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Bag of Words as CSV",
                data=csv,
                file_name='bag_of_words.csv',
                mime='text/csv',
            )
