#
# import nltk
#
# from PIL import Image
# import io
# import streamlit as st
# from poz_pp import data_preprocessing
# from poz_pp import NLP
#
# # Define Streamlit app title
#
# st.write(
#     f"""
#     <h1 style="color: #00FF00;">
#         <span style="font-weight: bold;">D</span>ART \U0001F3AF
#     </h1>
#     """,
#     unsafe_allow_html=True
# )
# st.write(
#     f"""
#     <h5 style="color: dark blue;">
#         Make your process more easier
#     </h5>
#     """,
#     unsafe_allow_html=True
# )
#
# # Sidebar for file upload
# uploaded_file = st.sidebar.file_uploader("Upload a file")
#
# # Read file content once uploaded
# # if uploaded_file is not None:
# #     file_content = uploaded_file.read().decode("utf-8")
#
# # Navigation bar for selecting section
# section = st.sidebar.radio("Navigation", ["Data Preprocessing", "NLP"])
#
# if section == "Data Preprocessing":
#     st.sidebar.subheader("Data Preprocessing")
#     preprocess_option = st.sidebar.selectbox("Select Data Preprocessing Option", (
#     "Handle Missing Values", "Transform Data", "Data Aggregation", "Split Data", "Handle Outliers"))
#
#     if uploaded_file is not None:
#         data = data_preprocessing.poz_read_file(uploaded_file)
#
#         # Display the uploaded data
#         # st.write("Uploaded Data:")
#         # st.write(data)
#
#         if preprocess_option == "Handle Missing Values":
#             try:
#                 fill_method = st.sidebar.selectbox("Select Fill Method", ("mean", "median", "mode", "constant"))
#                 df = data_preprocessing.poz_read_file(uploaded_file)
#                 if df is not None:
#                     st.write("Original Data:")
#                     st.write(df)
#
#                     filled_df = data_preprocessing.poz_handle_missing_values(df, fill_method)
#                     st.write("Data after handling missing values:")
#                     st.write(filled_df)
#             except Exception as e:
#                 st.error(f"An error occurred during handling missing values: {e}")
#
#         elif preprocess_option == "Transform Data":
#             try:
#                 df = data_preprocessing.poz_read_file(uploaded_file)
#                 if df is not None:
#                     st.write("Original Data:")
#                     st.write(df)
#
#                     transformation_method = st.sidebar.selectbox("Select Transformation Method",
#                                                                  ["Standardize", "Normalize", "Box-Cox", "Yeo-Johnson",
#                                                                   "Scaler", "Min-Max", "Log2"])
#                     transformed_df = data_preprocessing.poz_transformation(df,
#                                                                            method=transformation_method.lower().replace(
#                                                                                "-", ""))
#
#                     st.write(f"Data after {preprocess_option}:")
#                     st.write(transformed_df)
#             except Exception as e:
#                 st.error(f"An error occurred during data transformation: {e}")
#
#
#         elif preprocess_option == "Data Aggregation":
#             try:
#                 # Define grouping columns and aggregation functions
#                 group_by_columns = st.sidebar.multiselect("Select columns to group by:", data.columns)
#                 aggregation_functions = {}
#                 for column in data.columns:
#                     if column not in group_by_columns:
#                         aggregation_functions[column] = st.sidebar.selectbox(
#                             f"Select aggregation function for {column}:", ["mean", "sum", "count"])
#
#                 if st.sidebar.button("Aggregate Data"):
#                     aggregated_data = data_preprocessing.poz_data_aggregation(data, group_by_columns,
#                                                                               aggregation_functions)
#                     st.subheader("Aggregated Data:")
#                     st.write(aggregated_data)
#             except Exception as e:
#                 st.error(f"An error occurred during data aggregation setup: {e}")
#
#         elif preprocess_option == "Split Data":
#             try:
#                 target_column = st.sidebar.selectbox("Select the target column:", data.columns)
#                 test_size = st.sidebar.slider("Select the test size ratio:", 0.1, 0.5, 0.2)
#                 random_state = st.sidebar.number_input("Select random state:", value=42)
#                 axis = st.sidebar.radio("Select axis to drop target column:", [0, 1])
#
#                 if st.sidebar.button("Split Data"):
#                     X_train, X_test, y_train, y_test = data_preprocessing.poz_split_data(data, target_column, test_size,
#                                                                                          random_state, axis)
#                     st.subheader("Split Data:")
#                     st.write("X_train:", X_train)
#                     st.write("X_test:", X_test)
#                     st.write("y_train:", y_train)
#                     st.write("y_test:", y_test)
#             except Exception as e:
#                 st.error(f"An error occurred during data splitting: {e}")
#
#
#         elif preprocess_option == "Handle Outliers":
#             try:
#                 method = st.sidebar.selectbox("Select method for outlier detection:", ["z-score", "iqr"])
#                 threshold = st.sidebar.number_input("Enter threshold value:", value=3.0)
#                 action = st.sidebar.selectbox("Select action for handling outliers:", ["remove", "transform", "cap"])
#
#                 if st.sidebar.button("Handle Outliers"):
#                     cleaned_data = data_preprocessing.poz_outliers(data, method, threshold, action)
#                     st.subheader("Processed Data without Outliers:")
#                     st.write(cleaned_data)
#             except Exception as e:
#                 st.error(f"An error occurred during outlier handling: {e}")
#
#
# elif section == "NLP":
#     if uploaded_file is not None:
#         file_content = uploaded_file.read().decode("utf-8")
#     st.sidebar.subheader("NLP Tasks")
#     nlp_option = st.sidebar.selectbox("Select an NLP task:",
#                                       ["Tokenize Text", "Lowercase Text", "Remove Punctuation", "Stemming",
#                                        "Expand Contractions", "Remove Special Characters", "Remove Noisy Text",
#                                        "Word Embedding", "Padding", "Truncation", "Text Normalization", "Part-of-Speech Tagging"])
#
#     if nlp_option in ["Tokenize Text", "Lowercase Text", "Remove Punctuation", "Stemming", "Expand Contractions",
#                       "Remove Special Characters", "Remove Noisy Text", "Text Normalization"]:
#         st.subheader(nlp_option)
#         text = st.text_area("Enter text:", value=file_content)
#     else:
#         text = st.text_area("Enter text:", value=file_content)
#
#     if nlp_option == "Tokenize Text":
#         if st.button("Tokenize"):
#             tokens = NLP.poz_tokenize_text(text)
#             st.subheader("Tokens:")
#             st.write(tokens)
#             # Download button for tokens
#             if tokens:
#                 st.download_button(label="Download Tokens", data="\n".join(tokens), file_name="tokens.txt")
#
#     elif nlp_option == "Lowercase Text":
#         if st.button("Convert to Lowercase"):
#             lowercase_text = NLP.poz_lowercase_text(text)
#             st.subheader("Lowercased Text:")
#             st.write(lowercase_text)
#             # Download button for lowercase text
#             if lowercase_text:
#                 st.download_button(label="Download Lowercased Text", data=lowercase_text,
#                                    file_name="lowercase_text.txt")
#
#     elif nlp_option == "Remove Punctuation":
#         if st.button("Remove Punctuation"):
#             text_without_punctuation = NLP.poz_remove_punctuation(text)
#             st.subheader("Text without Punctuation:")
#             st.write(text_without_punctuation)
#             # Download button for text without punctuation
#             if text_without_punctuation:
#                 st.download_button(label="Download Text without Punctuation", data=text_without_punctuation,
#                                    file_name="text_without_punctuation.txt")
#
#     elif nlp_option == "Stemming":
#         if st.button("Stem"):
#             stemmed_text = NLP.poz_stemming_text(text)
#             st.subheader("Stemmed Text:")
#             st.write(stemmed_text)
#             # Download button for stemmed text
#             if stemmed_text:
#                 st.download_button(label="Download Stemmed Text", data=stemmed_text, file_name="stemmed_text.txt")
#
#     elif nlp_option == "Expand Contractions":
#         if st.button("Expand Contractions"):
#             expanded_text = NLP.poz_expand_contractions(text)
#             st.subheader("Expanded Text:")
#             st.write(expanded_text)
#             # Download button for expanded text
#             if expanded_text:
#                 st.download_button(label="Download Expanded Text", data=expanded_text, file_name="expanded_text.txt")
#
#     elif nlp_option == "Remove Special Characters":
#         if st.button("Remove Special Characters"):
#             text_without_special_chars = NLP.poz_remove_char(text)
#             st.subheader("Text without Special Characters:")
#             st.write(text_without_special_chars)
#             # Download button for text without special characters
#             if text_without_special_chars:
#                 st.download_button(label="Download Text without Special Characters", data=text_without_special_chars,
#                                    file_name="text_without_special_chars.txt")
#
#     elif nlp_option == "Remove Noisy Text":
#         if st.button("Remove Noisy Text"):
#             cleaned_text = NLP.poz_remove_noisy_text(text)
#             st.subheader("Cleaned Text:")
#             st.write(cleaned_text)
#             # Download button for cleaned text
#             if cleaned_text:
#                 st.download_button(label="Download Cleaned Text", data=cleaned_text, file_name="cleaned_text.txt")
#
#     elif nlp_option == "Word Embedding":
#         if st.button("Embed Words"):
#             word_embedding_model = NLP.poz_word_embed(text)
#             word_embedding_model.save("word_embedding.model")  # Save the Word2Vec model to a file
#
#             # Provide a download link to the Word2Vec model file
#             st.markdown(NLP.html("word_embedding.model", "Word Embedding Model"), unsafe_allow_html=True)
#
#     elif nlp_option == "Padding":
#         sequence = st.text_input("Enter sequence to pad:", value=text)
#         max_length = st.number_input("Enter max sequence length:")
#         if st.button("Pad Sequence"):
#             padded_sequence = NLP.poz_padding(sequence, max_length)
#             padded_sequence_str = '\n'.join(map(str, padded_sequence))  # Convert list to string
#             st.subheader("Padded Sequence:")
#             st.write(padded_sequence_str)
#             # Download button for padded sequence
#             if padded_sequence_str:
#                 st.download_button(label="Download Padded Sequence", data=padded_sequence_str,
#                                    file_name="padded_sequence.txt")
#
#     elif nlp_option == "Truncation":
#         sequence = st.text_input("Enter sequence to truncate:", value=text)
#         max_length = int(st.number_input("Enter max sequence length:"))  # Convert to integer
#         if st.button("Truncate Sequence"):
#             truncated_sequence = NLP.poz_truncation(sequence, max_length)
#             st.subheader("Truncated Sequence:")
#             st.write(truncated_sequence)
#             # Download button for truncated sequence
#             if truncated_sequence:
#                 st.download_button(label="Download Truncated Sequence", data=truncated_sequence,
#                                    file_name="truncated_sequence.txt")
#
#
#     elif nlp_option == "Text Normalization":
#         if st.button("Normalize Text"):
#             normalized_text = NLP.poz_nlp_normalize(text)
#             st.subheader("Normalized Text:")
#             st.write(normalized_text)
#             # Download button for normalized text
#             if normalized_text:
#                 st.download_button(label="Download Normalized Text", data=normalized_text,
#                                    file_name="normalized_text.txt")
#
#     # elif nlp_option == "Named Entity Recognition":
#     #     if st.button("Recognize Entities"):
#     #         entities = NLP.poz_name_entity_recognition(text)
#     #         st.subheader("Named Entities:")
#     #         st.write(entities)
#
#     #         # Convert entities to plain text
#     #         entities_text = "\n".join(entities)
#
#     #         # Download button for named entities
#     #         if entities:
#     #             st.download_button(label="Download Named Entities", data=entities_text, file_name="named_entities.txt")
#
#     elif nlp_option == "Part-of-Speech Tagging":
#         if st.button("Tag Parts of Speech"):
#             pos_tags = NLP.poz_part_of_speech(text)
#             st.subheader("Part-of-Speech Tags:")
#             st.write(pos_tags)
#             # Convert pos_tags to a string
#             pos_tags_str = '\n'.join([f"{word}\t{tag}" for word, tag in pos_tags])
#             # Download button for part-of-speech tags
#             if pos_tags_str:
#                 st.download_button(label="Download Part-of-Speech Tags", data=pos_tags_str, file_name="pos_tags.txt")
#
#
#
#
#
# from statistics import linear_regression
#
# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC, SVR
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import PolynomialFeatures
# from xgboost import XGBClassifier
# import numpy as np
#
# # Classification algorithms
#
# # Function to perform classification using Logistic Regression
# def logistic_regression(X_train, y_train, X_test, y_test):
#     lr_model = LinearRegression()
#     lr_model.fit(X_train, y_train)
#     y_pred = lr_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred.round())
#     precision = precision_score(y_test, y_pred.round())
#     recall = recall_score(y_test, y_pred.round())
#     f1 = f1_score(y_test, y_pred.round())
#     return accuracy, precision, recall, f1
#
# # Function to perform classification using Ridge Regression
# def ridge_regression(X_train, y_train, X_test, y_test):
#     ridge_model = Ridge()
#     ridge_model.fit(X_train, y_train)
#     y_pred = ridge_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred.round())
#     precision = precision_score(y_test, y_pred.round())
#     recall = recall_score(y_test, y_pred.round())
#     f1 = f1_score(y_test, y_pred.round())
#     return accuracy, precision, recall, f1
#
# # Function to perform classification using Lasso Regression
# def lasso_regression(X_train, y_train, X_test, y_test):
#     lasso_model = Lasso()
#     lasso_model.fit(X_train, y_train)
#     y_pred = lasso_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred.round())
#     precision = precision_score(y_test, y_pred.round())
#     recall = recall_score(y_test, y_pred.round())
#     f1 = f1_score(y_test, y_pred.round())
#     return accuracy, precision, recall, f1
#
# # Function to perform classification using Random Forest
# def random_forest(X_train, y_train, X_test, y_test):
#     rf_model = RandomForestClassifier()
#     rf_model.fit(X_train, y_train)
#     y_pred = rf_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     return accuracy, precision, recall, f1
#
# # Function to perform classification using Naive Bayes
# def naive_bayes(X_train, y_train, X_test, y_test):
#     nb_model = GaussianNB()
#     nb_model.fit(X_train, y_train)
#     y_pred = nb_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     return accuracy, precision, recall, f1
#
# # Function to perform classification using Decision Tree
# def decision_tree(X_train, y_train, X_test, y_test):
#     dt_model = DecisionTreeClassifier()
#     dt_model.fit(X_train, y_train)
#     y_pred = dt_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     return accuracy, precision, recall, f1
#
# # Function to perform regression using Linear Regression
# def linear_regression(X_train, y_train, X_test, y_test):
#     lr_model = LinearRegression()
#     lr_model.fit(X_train, y_train)
#     y_pred = lr_model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
#     return mse, mae, rmse, r2
#
#
# # Function to perform classification using AdaBoost
# def adaboost(X_train, y_train, X_test, y_test):
#     ada_model = AdaBoostClassifier()
#     ada_model.fit(X_train, y_train)
#     y_pred = ada_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     return accuracy, precision, recall, f1
#
# # Function to perform classification using Gradient Boosting
# def gradient_boosting(X_train, y_train, X_test, y_test):
#     gb_model = GradientBoostingClassifier()
#     gb_model.fit(X_train, y_train)
#     y_pred = gb_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     return accuracy, precision, recall, f1
#
# # Function to perform classification using XGBoost
# def xgboost(X_train, y_train, X_test, y_test):
#     xgb_model = XGBClassifier()
#     xgb_model.fit(X_train, y_train)
#     y_pred = xgb_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     return accuracy, precision, recall, f1
#
# # Function to perform regression using Polynomial Regression
# def polynomial_regression(X_train, y_train, X_test, y_test):
#     poly = PolynomialFeatures(degree=2)
#     X_train_poly = poly.fit_transform(X_train)
#     X_test_poly = poly.transform(X_test)
#     lr_model = LinearRegression()
#     lr_model.fit(X_train_poly, y_train)
#     y_pred = lr_model.predict(X_test_poly)
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
#     return mse, mae, rmse, r2
#
# # Main function
# def main():
#     st.title("Machine Learning Algorithms")
#     st.write("Upload your dataset in CSV format")
#
#     # Upload dataset
#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.write(df.head())
#
#         # Select task (classification or regression)
#         task = st.radio("Select Task", ("Classification", "Regression"))
#
#         # Select target column
#         target_column = st.selectbox("Select Target Column", options=df.columns)
#
#         # Split data into features and target
#         X = df.drop(columns=[target_column])
#         y = df[target_column]
#
#         # Split data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#         if task == "Classification":
#             # Perform classification with different algorithms
#             algorithms = {
#                 'Logistic Regression': logistic_regression,
#                 'Ridge Regression': ridge_regression,
#                 'Lasso Regression': lasso_regression,
#                 'Random Forest': random_forest,
#                 'Naive Bayes': naive_bayes,
#                 'Decision Tree': decision_tree,
#                 'AdaBoost': adaboost,
#                 'Gradient Boosting': gradient_boosting,
#                 'XGBoost': xgboost
#             }
#
#             st.write("Classification Performance Metrics:")
#             metrics_data = []
#             for name, algorithm in algorithms.items():
#                 accuracy, precision, recall, f1 = algorithm(X_train, y_train, X_test, y_test)
#                 metrics_data.append([name, accuracy, precision, recall, f1])
#
#             # Display classification performance metrics in a table
#             metrics_df = pd.DataFrame(metrics_data, columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
#             st.write(metrics_df)
#
#         elif task == "Regression":
#             # Perform regression with different algorithms
#             algorithms = {
#                 'Linear Regression': linear_regression,
#                 'Ridge Regression': ridge_regression,
#                 'Lasso Regression': lasso_regression,
#                 'Random Forest': random_forest,
#                 # 'Polynomial Regression': polynomial_regression
#             }
#
#             st.write("Regression Performance Metrics:")
#             metrics_data = []
#             for name, algorithm in algorithms.items():
#                 mse, mae, rmse, r2 = algorithm(X_train, y_train, X_test, y_test)
#                 metrics_data.append([name, mse, mae, rmse, r2])
#
#             # Display regression performance metrics in a table
#             metrics_df = pd.DataFrame(metrics_data, columns=['Algorithm', 'MSE', 'MAE', 'RMSE', 'R2 Score'])
#             st.write(metrics_df)
#
#  if __name__ == "__main__":
#     main()


import nltk
from PIL import Image
import io
import streamlit as st
from poz_pp import data_preprocessing
from poz_pp import NLP


# Add your logo image
logo = "ROR.png"

# Define Streamlit app title
st.sidebar.image(logo, use_column_width=True)

st.sidebar.write(
    f"""
    <h1 style="color: #00FF00;">
        <span style="font-weight: bold;">D</span>ART \U0001F3AF
    </h1>
    """,
    unsafe_allow_html=True
)
st.sidebar.write(
    f"""
    <h5 style="color: dark blue;">
        Make your process more easier
    </h5>
    """,
    unsafe_allow_html=True
)
styled_title_ml = """
    <h1 style="font-family: 'Peace Sans', sans-serif; letter-spacing: normal;">
        <span style="background: -webkit-linear-gradient(left,  #0000FF , #00FF00);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;">DART</span> \U0001F3AF<br>
    </h1>
"""
st.write(styled_title_ml, unsafe_allow_html=True)
# Navigation bar for selecting section
section = st.radio("Navigation", ["Data Preprocessing", "NLP"])

if section == "Data Preprocessing":
    gradient_text = '<span style="background: -webkit-linear-gradient(left,#0000FF , #00FF00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Data Preprocessing</span>'
    st.markdown(f"## {gradient_text}", unsafe_allow_html=True)
    preprocess_option = st.selectbox("Select Data Preprocessing Option", (
        "Handle Missing Values", "Transform Data", "Data Aggregation", "Split Data", "Handle Outliers"))

    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file is not None:
        data = data_preprocessing.poz_read_file(uploaded_file)

        if preprocess_option == "Handle Missing Values":
            try:
                fill_method = st.selectbox("Select Fill Method", ("mean", "median", "mode", "constant"))
                if data is not None:
                    st.write("Original Data:")
                    st.write(data)

                    filled_df = data_preprocessing.poz_handle_missing_values(data, fill_method)
                    st.write("Data after handling missing values:")
                    st.write(filled_df)
            except Exception as e:
                st.error(f"An error occurred during handling missing values: {e}")

        elif preprocess_option == "Transform Data":
            try:
                if data is not None:
                    st.write("Original Data:")
                    st.write(data)

                    transformation_method = st.selectbox("Select Transformation Method",
                                                          ["Standardize", "Normalize", "Box-Cox", "Yeo-Johnson",
                                                           "Scaler", "Min-Max", "Log2"])
                    transformed_df = data_preprocessing.poz_transformation(data,
                                                                           method=transformation_method.lower().replace(
                                                                               "-", ""))

                    st.write(f"Data after {preprocess_option}:")
                    st.write(transformed_df)
            except Exception as e:
                st.error(f"An error occurred during data transformation: {e}")

        elif preprocess_option == "Data Aggregation":
            try:
                # Define grouping columns and aggregation functions
                group_by_columns = st.multiselect("Select columns to group by:", data.columns)
                aggregation_functions = {}
                for column in data.columns:
                    if column not in group_by_columns:
                        aggregation_functions[column] = st.selectbox(
                            f"Select aggregation function for {column}:", ["mean", "sum", "count"])

                if st.button("Aggregate Data"):
                    aggregated_data = data_preprocessing.poz_data_aggregation(data, group_by_columns,
                                                                              aggregation_functions)
                    st.subheader("Aggregated Data:")
                    st.write(aggregated_data)
            except Exception as e:
                st.error(f"An error occurred during data aggregation setup: {e}")

        elif preprocess_option == "Split Data":
            try:
                target_column = st.selectbox("Select the target column:", data.columns)
                test_size = st.slider("Select the test size ratio:", 0.1, 0.5, 0.2)
                random_state = st.number_input("Select random state:", value=42)
                axis = st.radio("Select axis to drop target column:", [0, 1])

                if st.button("Split Data"):
                    X_train, X_test, y_train, y_test = data_preprocessing.poz_split_data(data, target_column, test_size,
                                                                                         random_state, axis)
                    st.subheader("Split Data:")
                    st.write("X_train:", X_train)
                    st.write("X_test:", X_test)
                    st.write("y_train:", y_train)
                    st.write("y_test:", y_test)
            except Exception as e:
                st.error(f"An error occurred during data splitting: {e}")

        elif preprocess_option == "Handle Outliers":
            try:
                method = st.selectbox("Select method for outlier detection:", ["z-score", "iqr"])
                threshold = st.number_input("Enter threshold value:", value=3.0)
                action = st.selectbox("Select action for handling outliers:", ["remove", "transform", "cap"])

                if st.button("Handle Outliers"):
                    cleaned_data = data_preprocessing.poz_outliers(data, method, threshold, action)
                    st.subheader("Processed Data without Outliers:")
                    st.write(cleaned_data)
            except Exception as e:
                st.error(f"An error occurred during outlier handling: {e}")

elif section == "NLP":
    gradient_text = '<span style="background: -webkit-linear-gradient(left,#0000FF , #00FF00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Natural Language Processing</span>'
    st.markdown(f"## {gradient_text}", unsafe_allow_html=True)
    nlp_option = st.selectbox("Select an NLP task:",
                              ["Tokenize Text", "Lowercase Text", "Remove Punctuation", "Stemming",
                               "Expand Contractions", "Remove Special Characters", "Remove Noisy Text",
                               "Word Embedding", "Padding", "Truncation", "Text Normalization", "Part-of-Speech Tagging"])

    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")

    if nlp_option in ["Tokenize Text", "Lowercase Text", "Remove Punctuation", "Stemming", "Expand Contractions",
                      "Remove Special Characters", "Remove Noisy Text", "Text Normalization"]:
        st.subheader(nlp_option)
        text = st.text_area("Enter text:", value=file_content)
    else:
        text = st.text_area("Enter text:", value=file_content)

    if nlp_option == "Tokenize Text":
        if st.button("Tokenize"):
            tokens = NLP.poz_tokenize_text(text)
            st.subheader("Tokens:")
            st.write(tokens)
            # Download button for tokens
            if tokens:
                st.download_button(label="Download Tokens", data="\n".join(tokens), file_name="tokens.txt")

    elif nlp_option == "Lowercase Text":
        if st.button("Convert to Lowercase"):
            lowercase_text = NLP.poz_lowercase_text(text)
            st.subheader("Lowercased Text:")
            st.write(lowercase_text)
            # Download button for lowercase text
            if lowercase_text:
                st.download_button(label="Download Lowercased Text", data=lowercase_text,
                                   file_name="lowercase_text.txt")

    elif nlp_option == "Remove Punctuation":
        if st.button("Remove Punctuation"):
            text_without_punctuation = NLP.poz_remove_punctuation(text)
            st.subheader("Text without Punctuation:")
            st.write(text_without_punctuation)
            # Download button for text without punctuation
            if text_without_punctuation:
                st.download_button(label="Download Text without Punctuation", data=text_without_punctuation,
                                   file_name="text_without_punctuation.txt")

    elif nlp_option == "Stemming":
        if st.button("Stem"):
            stemmed_text = NLP.poz_stemming_text(text)
            st.subheader("Stemmed Text:")
            st.write(stemmed_text)
            # Download button for stemmed text
            if stemmed_text:
                st.download_button(label="Download Stemmed Text", data=stemmed_text, file_name="stemmed_text.txt")

    elif nlp_option == "Expand Contractions":
        if st.button("Expand Contractions"):
            expanded_text = NLP.poz_expand_contractions(text)
            st.subheader("Expanded Text:")
            st.write(expanded_text)
            # Download button for expanded text
            if expanded_text:
                st.download_button(label="Download Expanded Text", data=expanded_text, file_name="expanded_text.txt")

    elif nlp_option == "Remove Special Characters":
        if st.button("Remove Special Characters"):
            text_without_special_chars = NLP.poz_remove_char(text)
            st.subheader("Text without Special Characters:")
            st.write(text_without_special_chars)
            # Download button for text without special characters
            if text_without_special_chars:
                st.download_button(label="Download Text without Special Characters", data=text_without_special_chars,
                                   file_name="text_without_special_chars.txt")

    elif nlp_option == "Remove Noisy Text":
        if st.button("Remove Noisy Text"):
            cleaned_text = NLP.poz_remove_noisy_text(text)
            st.subheader("Cleaned Text:")
            st.write(cleaned_text)
            # Download button for cleaned text
            if cleaned_text:
                st.download_button(label="Download Cleaned Text", data=cleaned_text, file_name="cleaned_text.txt")

    elif nlp_option == "Word Embedding":
        if st.button("Embed Words"):
            word_embedding_model = NLP.poz_word_embed(text)
            word_embedding_model.save("word_embedding.model")  # Save the Word2Vec model to a file

            # Provide a download link to the Word2Vec model file
            st.markdown(NLP.html("word_embedding.model", "Word Embedding Model"), unsafe_allow_html=True)

    elif nlp_option == "Padding":
        sequence = st.text_input("Enter sequence to pad:", value=text)
        max_length = st.number_input("Enter max sequence length:")
        if st.button("Pad Sequence"):
            padded_sequence = NLP.poz_padding(sequence, max_length)
            padded_sequence_str = '\n'.join(map(str, padded_sequence))  # Convert list to string
            st.subheader("Padded Sequence:")
            st.write(padded_sequence_str)
            # Download button for padded sequence
            if padded_sequence_str:
                st.download_button(label="Download Padded Sequence", data=padded_sequence_str,
                                   file_name="padded_sequence.txt")

    elif nlp_option == "Truncation":
        sequence = st.text_input("Enter sequence to truncate:", value=text)
        max_length = int(st.number_input("Enter max sequence length:"))  # Convert to integer
        if st.button("Truncate Sequence"):
            truncated_sequence = NLP.poz_truncation(sequence, max_length)
            st.subheader("Truncated Sequence:")
            st.write(truncated_sequence)
            # Download button for truncated sequence
            if truncated_sequence:
                st.download_button(label="Download Truncated Sequence", data=truncated_sequence,
                                   file_name="truncated_sequence.txt")

    elif nlp_option == "Text Normalization":
        if st.button("Normalize Text"):
            normalized_text = NLP.poz_nlp_normalize(text)
            st.subheader("Normalized Text:")
            st.write(normalized_text)
            # Download button for normalized text
            if normalized_text:
                st.download_button(label="Download Normalized Text", data=normalized_text,
                                   file_name="normalized_text.txt")

    elif nlp_option == "Part-of-Speech Tagging":
        if st.button("Tag Parts of Speech"):
            pos_tags = NLP.poz_part_of_speech(text)
            st.subheader("Part-of-Speech Tags:")
            st.write(pos_tags)
            # Convert pos_tags to a string
            pos_tags_str = '\n'.join([f"{word}\t{tag}" for word, tag in pos_tags])
            # Download button for part-of-speech tags
            if pos_tags_str:
                st.download_button(label="Download Part-of-Speech Tags", data=pos_tags_str, file_name="pos_tags.txt")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

# Streamlit Module
# import streamlit as st
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt


class FeatureEngineering:

    @staticmethod
    def feature_extraction(X, feature_names, y=None, method='Principal Component Analysis (PCA)', n_components=2, k=5):
        if method == 'Principal Component Analysis (PCA)':
            pca = PCA(n_components=n_components)
            X_transformed = pca.fit_transform(X)
            columns = [f"PC{i + 1}" for i in range(n_components)]
        elif method == 'scaling':
            scaler = MinMaxScaler()
            X_transformed = scaler.fit_transform(X)
            columns = feature_names
        elif method == 'polynomial':
            poly = PolynomialFeatures(degree=2, include_bias=False)  # Adjust degree as needed
            X_transformed = poly.fit_transform(X)
            # Generate feature names
            poly_feature_names = poly.get_feature_names_out(input_features=feature_names)
            columns = [name.replace(" ", "*") for name in poly_feature_names]  # Replace spaces with *
        elif method == 'selectkbest':
            selector = SelectKBest(score_func=f_classif, k=k)
            X_transformed = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            columns = [feature_names[i] for i in selected_indices]
        else:
            raise ValueError(
                "Invalid method. Choose one of: 'Principal Component Analysis (PCA)', 'scaling', 'polynomial', 'selectkbest'")

        return pd.DataFrame(X_transformed, columns=columns)

    def poz_feature_scaling(data, method='standardize'):
        if method == 'normalize':
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            scaled_data = (data - min_val) / (max_val - min_val)
        elif method == 'standardize':
            mean = np.mean(data, axis=0)
            std_dev = np.std(data, axis=0)
            scaled_data = (data - mean) / std_dev
        elif method == 'clip_log_scale':
            clipped_data = np.clip(data, 0, 1)
            scaled_data = np.log(clipped_data + 1)
        elif method == 'z_score':
            mean = np.mean(data, axis=0)
            std_dev = np.std(data, axis=0)
            scaled_data = (data - mean) / std_dev
        else:
            raise ValueError(
                "Invalid method. Options: 'normalize', 'standardize', 'clip_log_scale', 'z_score', 'robust'")
        return scaled_data

    def plot_feature_importances(X, y, feature_names, algorithm='Random Forest', n_estimators_rf=100,
                                 n_estimators_xgb=100, random_state=42):
        """
        Plot feature importances using Random Forest or XGBoost classifiers.

        Parameters:
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels).
        feature_names : list
            List of feature names.
        algorithm : str, default='Random Forest'
            The algorithm to use for feature importance calculation. It can be 'Random Forest' or 'XGBoost'.
        n_estimators_rf : int, default=100
            The number of trees in the Random Forest.
        n_estimators_xgb : int, default=100
            The number of boosting rounds in XGBoost.
        random_state : int or RandomState, default=42
            Controls both the randomness of the bootstrapping of the samples
            used when building trees and the sampling of the features to consider
            when looking for the best split at each node.

        Returns:
        None
        """
        if algorithm == 'Random Forest':
            # Train a Random Forest classifier
            clf = RandomForestClassifier(n_estimators=n_estimators_rf, random_state=random_state)
        elif algorithm == 'XGBoost':
            # Train an XGBoost classifier
            clf = xgb.XGBClassifier(n_estimators=n_estimators_xgb, random_state=random_state)
        else:
            st.error("Invalid algorithm selected.")
            return

        clf.fit(X, y)

        # Get feature importances
        importances = clf.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        st.write("Feature ranking:")
        for f in range(X.shape[1]):
            st.write(f"{f + 1}. Feature {feature_names[indices[f]]} ({importances[indices[f]]})")

        # Plot the feature importances
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Feature importance")
        st.pyplot(plt)


def main():
    styled_title_ml = """
        <h1 style="font-family: 'Peace Sans', sans-serif; letter-spacing: normal;">
            <span style="background: -webkit-linear-gradient(left,#0000FF , #00FF00);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">FEATURE ENGINEERING</span><br>
        </h1>
    """
    st.write(styled_title_ml, unsafe_allow_html=True)


    # File uploader for data upload
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.write(data.head())

        feature_names = data.columns.tolist()

        # Feature scaling
        st.header("Feature Scaling")
        scaling_method = st.selectbox("Select feature scaling method:",
                                      ["Standardize", "Normalize", "Clip Log Scale", "Z-score"])
        if scaling_method:
            if scaling_method == "Standardize":
                scaled_data = FeatureEngineering.poz_feature_scaling(data, method='standardize')
            elif scaling_method == "Normalize":
                scaled_data = FeatureEngineering.poz_feature_scaling(data, method='normalize')
            elif scaling_method == "Clip Log Scale":
                scaled_data = FeatureEngineering.poz_feature_scaling(data, method='clip_log_scale')
            elif scaling_method == "Z-score":
                scaled_data = FeatureEngineering.poz_feature_scaling(data, method='z_score')

            st.write("Scaled data:")
            st.write(scaled_data.head())

        # Feature extraction
        st.header("Feature Extraction")
        method = st.selectbox("Select feature extraction method:",
                              ["Principal Component Analysis (PCA)", "Polynomial Features", "SelectKBest"])
        if method:
            if method == "Principal Component Analysis (PCA)":
                n_components = st.slider("Number of components:", min_value=2, max_value=len(feature_names), value=2)
                transformed_data = FeatureEngineering.feature_extraction(data, feature_names, method=method,
                                                                         n_components=n_components)
            elif method == "Polynomial Features":
                transformed_data = FeatureEngineering.feature_extraction(data, feature_names, method='polynomial')
            elif method == "SelectKBest":
                k = st.slider("Number of top features to select:", min_value=1, max_value=len(feature_names) - 1,
                              value=2)  # -1 to exclude target column
                target = st.selectbox("Select target column:", feature_names)
                y = data[target]
                transformed_data = FeatureEngineering.feature_extraction(data.drop(columns=[target]),
                                                                         feature_names[:-1], y=y, method=method.lower(),
                                                                         k=k)  # <-- Corrected method name

            st.write("Transformed data:")
            st.write(transformed_data.head())

        # Plot feature importances
        st.header("Feature Importance")
        algorithm = st.selectbox("Select algorithm for feature importance calculation:", ["Random Forest", "XGBoost"],
                                 key="algorithm_selectbox")
        target = st.selectbox("Select target column:", feature_names, key="target_column_selectbox")
        if algorithm:
            if st.button("Plot Feature Importances"):
                y = data[target]
                FeatureEngineering.plot_feature_importances(data.drop(columns=[target]), y, feature_names[:-1],
                                                            algorithm=algorithm)


if __name__ == "__main__":
    main()

from statistics import linear_regression

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier
import numpy as np

# Classification algorithms

# Function to perform classification using Logistic Regression
def logistic_regression(X_train, y_train, X_test, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round())
    recall = recall_score(y_test, y_pred.round())
    f1 = f1_score(y_test, y_pred.round())
    return accuracy, precision, recall, f1

# Function to perform classification using Ridge Regression
def ridge_regression(X_train, y_train, X_test, y_test):
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round())
    recall = recall_score(y_test, y_pred.round())
    f1 = f1_score(y_test, y_pred.round())
    return accuracy, precision, recall, f1

# Function to perform classification using Lasso Regression
def lasso_regression(X_train, y_train, X_test, y_test):
    lasso_model = Lasso()
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round())
    recall = recall_score(y_test, y_pred.round())
    f1 = f1_score(y_test, y_pred.round())
    return accuracy, precision, recall, f1

# Function to perform classification using Random Forest
def random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Function to perform classification using Naive Bayes
def naive_bayes(X_train, y_train, X_test, y_test):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Function to perform classification using Decision Tree
def decision_tree(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Function to perform regression using Linear Regression
def linear_regression(X_train, y_train, X_test, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, rmse, r2


# Function to perform classification using AdaBoost
def adaboost(X_train, y_train, X_test, y_test):
    ada_model = AdaBoostClassifier()
    ada_model.fit(X_train, y_train)
    y_pred = ada_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Function to perform classification using Gradient Boosting
def gradient_boosting(X_train, y_train, X_test, y_test):
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)
    y_pred = gb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Function to perform classification using XGBoost
def xgboost(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Function to perform regression using Polynomial Regression
def polynomial_regression(X_train, y_train, X_test, y_test):
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    lr_model = LinearRegression()
    lr_model.fit(X_train_poly, y_train)
    y_pred = lr_model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, rmse, r2

# Main function
def main():
    styled_title_ml = """
        <h1 style="font-family: 'Peace Sans', sans-serif; letter-spacing: normal;">
            <span style="background: -webkit-linear-gradient(left, #0000FF , #00FF00);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">ML-Pipeline</span><br>
        </h1>
    """
    st.write(styled_title_ml, unsafe_allow_html=True)



    st.write("Upload your dataset in CSV format")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        # Select task (classification or regression)
        task = st.radio("Select Task", ("Classification", "Regression"))

        # Select target column
        target_column = st.selectbox("Select Target Column", options=df.columns)

        # Split data into features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task == "Classification":
            # Perform classification with different algorithms
            algorithms = {
                'Logistic Regression': logistic_regression,
                'Ridge Regression': ridge_regression,
                'Lasso Regression': lasso_regression,
                'Random Forest': random_forest,
                'Naive Bayes': naive_bayes,
                'Decision Tree': decision_tree,
                'AdaBoost': adaboost,
                'Gradient Boosting': gradient_boosting,
                'XGBoost': xgboost
            }

            st.write("Classification Performance Metrics:")
            metrics_data = []
            for name, algorithm in algorithms.items():
                accuracy, precision, recall, f1 = algorithm(X_train, y_train, X_test, y_test)
                metrics_data.append([name, accuracy, precision, recall, f1])

            # Display classification performance metrics in a table
            metrics_df = pd.DataFrame(metrics_data, columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
            st.write(metrics_df)

        elif task == "Regression":
            # Perform regression with different algorithms
            algorithms = {
                'Linear Regression': linear_regression,
                'Ridge Regression': ridge_regression,
                'Lasso Regression': lasso_regression,
                'Random Forest': random_forest,
                # 'Polynomial Regression': polynomial_regression
            }

            st.write("Regression Performance Metrics:")
            metrics_data = []
            for name, algorithm in algorithms.items():
                mse, mae, rmse, r2 = algorithm(X_train, y_train, X_test, y_test)
                metrics_data.append([name, mse, mae, rmse, r2])

            # Display regression performance metrics in a table
            metrics_df = pd.DataFrame(metrics_data, columns=['Algorithm', 'MSE', 'MAE', 'RMSE', 'R2 Score'])
            st.write(metrics_df)

if __name__ == "__main__":
    main()



