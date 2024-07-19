# com.py
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import streamlit as st 

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
    return ' '.join(words)

# Function to compute Bag of Words
def compute_bag_of_words(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df

# Function to generate Bag-of-Words features
def generate_word2vec_features(corpus, size=100, window=5, min_count=1):
    tokenized_corpus = [preprocess_text(doc) for doc in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=size, window=window, min_count=min_count, workers=4)
    
    # Get the average word vectors for each document
    def get_average_word2vec(tokens_list, model, size):
        feature_vec = np.zeros((size,), dtype="float32")
        n_words = 0
        for token in tokens_list:
            if token in model.wv.key_to_index:
                n_words += 1
                feature_vec = np.add(feature_vec, model.wv[token])
        if n_words > 0:
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    word2vec_features = np.array([get_average_word2vec(tokens, model, size) for tokens in tokenized_corpus])
    return pd.DataFrame(word2vec_features), model

# Function to train CBOW model
def train_cbow(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    return model

# Function to train Skipgram model
def train_skipgram_model(sentences):
    st.subheader("Skipgram Model Training...")
    tokenized_data = [preprocess_text(doc) for doc in sentences]
    model = Word2Vec(sentences=tokenized_data, vector_size=50, window=4, sg=1, min_count=1)
    model.train(tokenized_data, total_examples=len(tokenized_data), epochs=10)
    st.success("Skipgram Model Trained Successfully!")
    
    embeddings = []
    words = []
    
    for sentence in tokenized_data:
        for word in sentence:
            try:
                embedding_vector = model.wv[word]
                embeddings.append(embedding_vector)
                words.append(word)
            except KeyError:
                st.warning(f"Word '{word}' not found in the vocabulary.")
    
    if embeddings:
        embedding_df = pd.DataFrame(embeddings, index=words)
        st.subheader("Embedding Vectors for All Words:")
        st.write(embedding_df)

# Function to compute N-grams and TF-IDF
def compute_tfidf(corpus, ngram_range=(1, 2), top_n=10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)
    tfidf_scores = X.sum(axis=0).A1
    tfidf_scores_df = pd.DataFrame({
        'ngram': vectorizer.get_feature_names_out(),
        'tfidf': tfidf_scores
    })
    top_ngrams = tfidf_scores_df.sort_values(by='tfidf', ascending=False).head(top_n)
    return top_ngrams
