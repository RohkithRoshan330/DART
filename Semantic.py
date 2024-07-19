# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#
# # Streamlit application
# st.title("Email Spam Detection")
#
# st.write("This application classifies emails as spam or not spam using a trained machine learning model.")
#
# # File uploader to upload the dataset
# uploaded_file = st.file_uploader("Upload your email dataset (CSV format)", type="csv")
#
# if uploaded_file is not None:
#     try:
#         # Load the dataset
#         data = pd.read_csv(uploaded_file)
#
#         # Display the first few rows of the dataset
#         st.write("Dataset Preview:")
#         st.write(data.head())
#
#         # Ask the user to select the text and label columns
#         text_column = st.selectbox("Select the column containing the email text:", data.columns)
#         label_column = st.selectbox("Select the column containing the labels:", data.columns)
#
#         # Extract the selected columns
#         X = data[text_column]
#         y = data[label_column]
#
#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#         # Create a pipeline that combines the TfidfVectorizer with a MultinomialNB classifier
#         model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#
#         # Train the model
#         model.fit(X_train, y_train)
#
#         # Predict on the test set
#         y_pred = model.predict(X_test)
#
#         # Evaluate the model
#         accuracy = accuracy_score(y_test, y_pred)
#         conf_matrix = confusion_matrix(y_test, y_pred)
#         class_report = classification_report(y_test, y_pred)
#
#         st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
#
#         st.subheader("Confusion Matrix")
#         st.write(conf_matrix)
#
#         st.subheader("Classification Report")
#         st.text(class_report)
#
#         # User input for email text
#         email_text = st.text_area("Enter the email text:")
#
#         if st.button("Predict"):
#             # Predict if the input email text is spam or not
#             prediction = model.predict([email_text])[0]
#             if prediction == 'spam':
#                 st.error("This email is spam.")
#             else:
#                 st.success("This email is not spam.")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
# else:
#     st.info("Please upload a CSV file to proceed.")









import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Streamlit application
st.title("Text Classification")

st.write("This application classifies emails as spam or not spam using a trained machine learning model.")

# File uploader to upload the dataset
uploaded_file = st.file_uploader("Upload your email dataset (CSV format)", type="csv")

if uploaded_file is not None:
    try:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Display the first few rows of the dataset
        st.write("Dataset Preview:")
        st.write(data.head())

        # Ask the user to select the text and label columns
        text_column = st.selectbox("Select the column containing the email text:", data.columns)
        label_column = st.selectbox("Select the column containing the labels:", data.columns)

        # Extract the selected columns and preprocess the text
        X = data[text_column].apply(preprocess_text)
        y = data[label_column]

        # Split the data into training and testing sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Create a pipeline that combines the TfidfVectorizer with a LogisticRegression
        pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

        # Define hyperparameters for GridSearchCV
        parameters = {
            'tfidfvectorizer__max_df': [0.75, 1.0],
            'tfidfvectorizer__min_df': [1, 2],
            'logisticregression__C': [0.1, 1.0, 10.0]
        }

        # Use StratifiedKFold for handling class imbalance
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Perform GridSearchCV to find the best parameters
        grid_search = GridSearchCV(pipeline, parameters, cv=stratified_kfold, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Predict on the test set
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        st.subheader("Confusion Matrix")
        st.write(conf_matrix)

        st.subheader("Classification Report")
        st.text(class_report)

        # User input for email text
        email_text = st.text_area("Enter the email text:")

        if st.button("Predict"):
            # Predict if the input email text is spam or not
            processed_text = preprocess_text(email_text)
            prediction = best_model.predict([processed_text])[0]
            if prediction == 'spam':
                st.error("This email is spam.")
            else:
                st.success("This email is not spam.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
