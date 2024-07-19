import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize text
    words = [word for word in words if word.isalpha()]  # Remove non-alphabetic tokens
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return words

# Function to train CBOW model
def train_cbow(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    return model

# Streamlit app
st.title("CBOW Embeddings Generator")

# Options for input method
input_method = st.radio("Choose input method:", ("Upload File", "Enter Paragraph"))

if input_method == "Upload File":
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

        if st.button("Train CBOW Model"):
            with st.spinner("Training CBOW model..."):
                # Train CBOW model
                cbow_model = train_cbow(sentences)
                st.success("CBOW model trained successfully!")

                # Display embeddings for a few words
                words = cbow_model.wv.index_to_key[:10]  # Get the first 10 words
                embeddings = {word: cbow_model.wv[word] for word in words}

                st.write("CBOW Embeddings for the first 10 words:")
                st.write(embeddings)

                # Show a downloadable CSV with the embeddings
                embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
                csv = embeddings_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download embeddings as CSV",
                    data=csv,
                    file_name='cbow_embeddings.csv',
                    mime='text/csv',
                )

elif input_method == "Enter Paragraph":
    paragraph = st.text_area("Enter a paragraph of text")

    if paragraph:
        sentences = [preprocess_text(paragraph)]

        if st.button("Train CBOW Model"):
            with st.spinner("Training CBOW model..."):
                # Train CBOW model
                cbow_model = train_cbow(sentences)
                st.success("CBOW model trained successfully!")

                # Display embeddings for a few words
                words = cbow_model.wv.index_to_key[:10]  # Get the first 10 words
                embeddings = {word: cbow_model.wv[word] for word in words}

                st.write("CBOW Embeddings for the first 10 words:")
                st.write(embeddings)

                # Show a downloadable CSV with the embeddings
                embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
                csv = embeddings_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download embeddings as CSV",
                    data=csv,
                    file_name='cbow_embeddings.csv',
                    mime='text/csv',
                )
