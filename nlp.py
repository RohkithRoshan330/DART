# Modules for nlp
import nltk
import string
import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import base64

# NLP Package Installation
nltk.download('punk')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


class NLP:

    @staticmethod
    def poz_tokenize_text(text):
        import nltk
        nltk.download('punkt')
        tokens = word_tokenize(text)
        return tokens

    @staticmethod
    def poz_lowercase_text(text):
        # Convert text to lowercase using the lower() method
        lowercased_text = text.lower()
        return lowercased_text

    @staticmethod
    def poz_remove_punctuation(text):
        # Define a translation table that maps all punctuation characters to None
        translation_table = str.maketrans('', '', string.punctuation)
        # Remove punctuation using the translation table
        text_without_punctuation = text.translate(translation_table)
        return text_without_punctuation

    @staticmethod
    def poz_stemming_text(text):
        # Download NLTK resources if not already downloaded
        nltk.download('punkt', quiet=True)
        # Initialize Porter Stemmer
        stemmer = PorterStemmer()
        # Tokenize the input text string
        words = word_tokenize(text)
        # Stem each word in the tokenized words
        stemmed_words = [stemmer.stem(word) for word in words]
        # Join the stemmed words back into a string
        stemmed_text = ' '.join(stemmed_words)
        return stemmed_text

    @staticmethod
    def poz_expand_contractions(text):
        contractions_dict = {
            "ain't": "am not / are not / is not / has not / have not",
            "aren't": "are not / am not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he had / he would",
            "he'd've": "he would have",
            "he'll": "he shall / he will",
            "he'll've": "he shall have / he will have",
            "he's": "he has / he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how has / how is / how does",
            "I'd": "I had / I would",
            "I'd've": "I would have",
            "I'll": "I shall / I will",
            "I'll've": "I shall have / I will have",
            "I'm": "I am",
            "I've": "I have",
            "isn't": "is not",
            "it'd": "it had / it would",
            "it'd've": "it would have",
            "it'll": "it shall / it will",
            "it'll've": "it shall have / it will have",
            "it's": "it has / it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she had / she would",
            "she'd've": "she would have",
            "she'll": "she shall / she will",
            "she'll've": "she shall have / she will have",
            "she's": "she has / she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so as / so is",
            "that'd": "that would / that had",
            "that'd've": "that would have",
            "that's": "that has / that is",
            "there'd": "there had / there would",
            "there'd've": "there would have",
            "there's": "there has / there is",
            "they'd": "they had / they would",
            "they'd've": "they would have",
            "they'll": "they shall / they will",
            "they'll've": "they shall have / they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had / we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what shall / what will",
            "what'll've": "what shall have / what will have",
            "what're": "what are",
            "what's": "what has / what is",
            "what've": "what have",
            "when's": "when has / when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where has / where is",
            "where've": "where have",
            "who'll": "who shall / who will",
            "who'll've": "who shall have / who will have",
            "who's": "who has / who is",
            "who've": "who have",
            "why's": "why has / why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you had / you would",
            "you'd've": "you would have",
            "you'll": "you shall / you will",
            "you'll've": "you shall have / you will have",
            "you're": "you are",
            "you've": "you have",
        }
        # Split the text into words
        words = text.split()
        # Expand contractions
        expanded_words = [contractions_dict.get(word, word) for word in words]
        # Reconstruct the text
        expanded_text = " ".join(expanded_words)
        return expanded_text

    @staticmethod
    def poz_remove_char(text):
        # Remove numerical digits and special characters
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'[^\w\s\_]', '', text)  # Remove special characters
        return text

    @staticmethod
    def remove_noisy_text(text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove other noisy text patterns as needed
        return text

    @staticmethod
    def poz_remove_noisy_text(text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove other noisy text patterns as needed
        return text

    @staticmethod
    def poz_word_embed(text):
        # Tokenize the text into words
        words = word_tokenize(text)
        # Train Word2Vec model
        model = Word2Vec([words], vector_size=100, window=5, min_count=1, workers=4).wv
        return model

    @staticmethod
    def html(file_path, file_label):
        with open(file_path, 'rb') as file:
            file_content = file.read()
        encoded_file = base64.b64encode(file_content).decode()
        href = f'<a href="data:file/txt;base64,{encoded_file}" download="{file_label}">Click here to download {file_label}</a>'
        return href

    @staticmethod
    def poz_padding(sequence, max_length):
        if isinstance(sequence, str):  # Check if sequence is a string
            sequence = sequence.split()  # Convert string to list of tokens
        max_length = int(max_length)
        padded_sequence = sequence[:max_length] + [0] * (max_length - len(sequence))
        return padded_sequence

    @staticmethod
    def poz_truncation(sequence, max_length):
        truncated_sequence = sequence[:max_length]
        return truncated_sequence

    @staticmethod
    def poz_nlp_normalize(text):
        normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return normalized_text

    @staticmethod
    def poz_name_entity_recognition(text):
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Perform part-of-speech tagging
        pos_tags = nltk.pos_tag(tokens)
        # Perform named entity recognition
        entities = nltk.ne_chunk(pos_tags)
        return entities

    @staticmethod
    def poz_part_of_speech(text):
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Perform part-of-speech tagging
        pos_tags = nltk.pos_tag(tokens)
        return pos_tags