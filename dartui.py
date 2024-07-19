import nltk
from PIL import Image
import io
import streamlit as st
from data_preprocesing import data_preprocessing
from mmm import load_and_predict_model, train_models
from nlp import NLP
from featureengineering import FeatureEngineering
from featureengineering import FeatureSelection
import matplotlib.pyplot as plt
import seaborn as sns
from imbalance_data import Imbalance
from mlpipeline import MLPIPE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


# import tensorflow as tf
# from tensorflow.keras import layers
from collections import Counter

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

        # if preprocess_option == "Handle Missing Values":
        #     try:
        #         fill_method = st.selectbox("Select Fill Method", ("mean", "median", "mode", "constant"))
        #         if data is not None:
        #             st.write("Original Data:")
        #             st.write(data)
        #
        #             filled_df = data_preprocessing.poz_handle_missing_values(data, fill_method)
        #             st.write("Data after handling missing values:")
        #             st.write(filled_df)
        #     except Exception as e:
        #         st.error(f"An error occurred during handling missing values: {e}")
        if preprocess_option == "Handle Missing Values":
            try:
                fill_method = st.selectbox("Select Fill Method",
                                           ("mean", "median", "mode", "constant", "linear_regression", "random_forest"))
                if data is not None:
                    st.write("Original Data:")
                    st.write(data)

                    if fill_method in ["mean", "median", "mode", "constant"]:
                        filled_df = data_preprocessing.poz_handle_missing_values(data, fill_method)
                    elif fill_method == "linear_regression":
                        filled_df = data_preprocessing.fill_missing_values_linear(data)
                    elif fill_method == "random_forest":
                        filled_df = data_preprocessing.fill_missing_values_rf(data)

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
                action = st.selectbox("Select action for handling outliers:", ["remove", "cap"])

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
                               "Word Embedding", "Padding", "Truncation", "Text Normalization",
                               "Part-of-Speech Tagging"])

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


styled_title_ml = """
    <h1 style="font-family: 'Peace Sans', sans-serif; letter-spacing: normal;">
        <span style="background: -webkit-linear-gradient(left,#0000FF , #00FF00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;">Feature Engineering</span><br>
    </h1>
"""
st.write(styled_title_ml, unsafe_allow_html=True)

# Add new section for Feature Selection
section = st.selectbox("Select a section:",
                       ["Feature Scaling", "Feature Extraction", "Plot Feature Importance", "Feature Selection"])

if section == "Feature Scaling":
    # Existing Feature Scaling code
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.write(data.head())

        feature_names = data.columns.tolist()

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

elif section == "Feature Extraction":
    # Existing Feature Extraction code
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.write(data.head())

        feature_names = data.columns.tolist()

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

elif section == "Plot Feature Importance":
    # Existing Plot Feature Importance code
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.write(data.head())

        feature_names = data.columns.tolist()

        algorithm = st.selectbox("Select algorithm for feature importance calculation:", ["Random Forest", "XGBoost"])
        target = st.selectbox("Select target column:", feature_names)

        if algorithm:
            if st.button("Plot Feature Importances"):
                y = data[target]
                FeatureEngineering.plot_feature_importances(data.drop(columns=[target]), y, feature_names[:-1],
                                                            algorithm=algorithm)

elif section == "Feature Selection":
    # New Feature Selection code
    uploaded_file = st.file_uploader("Upload your CSV data:", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Preprocess the dataset
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        le = LabelEncoder()
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = le.fit_transform(df[column])

        processing_method = st.selectbox("Select Processing Method:",
                                         ("Filter Method", "Wrapping Method", "Embedding Method"))

        selected_features = []
        feature_scores = pd.DataFrame()
        accuracy_after_fs = 0

        if processing_method == "Filter Method":
            selected_features, feature_scores = FeatureSelection.filter_method(df)
        elif processing_method == "Wrapping Method":
            target_column = st.selectbox("Select the target column", df.columns)
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.30)

            st.subheader("Feature Selection")
            method = st.selectbox("Select Feature Selection Method",
                                  ["Forward Selection", "Backward Elimination", "Recursive Feature Elimination"])
            n_features_limit = len(X.columns)
            n_features = st.slider("Select number of features to keep", min_value=1, max_value=n_features_limit,
                                   value=5)

            if method == "Forward Selection":
                support, indices = FeatureSelection.forward_selection(X_train, y_train, n_features)
            elif method == "Backward Elimination":
                support, indices = FeatureSelection.backward_elimination(X_train, y_train, n_features)
            elif method == "Recursive Feature Elimination":
                support, ranking = FeatureSelection.recursive_feature_elimination(X_train, y_train, n_features)
                indices = [i for i, x in enumerate(support) if x]

            selected_features = X.columns[indices]
            st.write("Selected Features:", selected_features.tolist())

            X_train_fs = X_train[selected_features]
            X_test_fs = X_test[selected_features]

            st.subheader("Classification")
            algorithm = st.selectbox("Select algorithm", ["Random Forest", "XGBoost", "Gradient Boosting"])

            if algorithm == "Random Forest":
                model = RandomForestClassifier(random_state=100, n_estimators=50)
            elif algorithm == "XGBoost":
                model = xgb.XGBClassifier(random_state=100, n_estimators=50)
            elif algorithm == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=100, n_estimators=50)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_before_fs = metrics.accuracy_score(y_test, y_pred)
            st.write(f"Accuracy before feature selection: {accuracy_before_fs}")

            model.fit(X_train_fs, y_train)
            y_pred_fs = model.predict(X_test_fs)
            accuracy_after_fs = metrics.accuracy_score(y_test, y_pred_fs)
            st.write(f"Accuracy after feature selection: {accuracy_after_fs}")

            importances = model.feature_importances_
            final_df = pd.DataFrame({"Features": selected_features, "Importances": importances})
            final_df.set_index('Features', inplace=True)
            final_df = final_df.sort_values('Importances')

            st.subheader("Feature Importances after Feature Selection")
            plt.figure(figsize=(10, 3))
            plt.xticks(rotation=45)
            sns.barplot(x="Features", y="Importances", data=final_df.reset_index())
            st.pyplot(plt)
        elif processing_method == "Embedding Method":
            selected_features, feature_scores = FeatureSelection.embedding_method(df)

            target_column = st.selectbox("Select the target column", df.columns)
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.30)
            X_train_fs = X_train[selected_features]
            X_test_fs = X_test[selected_features]

            st.subheader("Classification")
            algorithm = st.selectbox("Select algorithm", ["Random Forest", "XGBoost", "Gradient Boosting"])

            if algorithm == "Random Forest":
                model = RandomForestClassifier(random_state=100, n_estimators=50)
            elif algorithm == "XGBoost":
                model = xgb.XGBClassifier(random_state=100, n_estimators=50)
            elif algorithm == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=100, n_estimators=50)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_before_fs = metrics.accuracy_score(y_test, y_pred)
            st.write(f"Accuracy before feature selection: {accuracy_before_fs}")

            model.fit(X_train_fs, y_train)
            y_pred_fs = model.predict(X_test_fs)
            accuracy_after_fs = metrics.accuracy_score(y_test, y_pred_fs)
            st.write(f"Accuracy after feature selection: {accuracy_after_fs}")

        if not feature_scores.empty and accuracy_after_fs:
            feature_scores['importance_rate'] = feature_scores['score'] / feature_scores['score'].sum()
            feature_scores['accuracy_rate'] = feature_scores['importance_rate'] * accuracy_after_fs

            st.subheader("Feature Importance in Accuracy Rate Table:")
            st.dataframe(feature_scores[['feature', 'score', 'importance_rate', 'accuracy_rate']])

            st.subheader("Feature Importances Graph")
            plt.figure(figsize=(10, 3))
            plt.xticks(rotation=45)
            sns.barplot(x="feature", y="importance_rate", data=feature_scores.reset_index())
            st.pyplot(plt)
    else:
        st.write("Please upload a CSV file.")

styled_title_ml = """
    <h1 style="font-family: 'Peace Sans', sans-serif; letter-spacing: normal;">
        <span style="background: -webkit-linear-gradient(left,#0000FF , #00FF00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;">Handling Imbalance Dataset</span><br>
    </h1>
"""
st.write(styled_title_ml, unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your CSV data:", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        st.write("Dataset:")
        st.write(df.head())

        target_column = st.selectbox("Select the target column:", df.columns, index=0)
        method = st.selectbox("Select resampling method:",
                              ['under', 'over', 'SMOTE', 'ADASYN', 'Borderline-SMOTE', 'Synthetic Data Generation'])
        sampling_strategy = 'auto'
        if method in ['under', 'over']:
            sampling_strategy = st.text_input("Enter sampling strategy (leave 'auto' for default):", 'auto')

        if st.button("Resample Dataset"):
            X = df.drop(columns=[target_column])
            y = df[target_column]

            st.write("Class distribution before resampling:")
            st.write(Counter(y))
            Imbalance.plot_class_distribution(y, "Original Dataset Class Distribution")

            if method in ['under', 'over', 'SMOTE', 'ADASYN', 'Borderline-SMOTE']:
                X_res, y_res = Imbalance.balance_dataset(X, y, method=method, sampling_strategy=sampling_strategy)
            elif method == 'Synthetic Data Generation':
                num_rows = st.number_input("Enter the number of synthetic rows to generate:", min_value=1, value=5,
                                           step=1)
                synthetic_data = Imbalance.generate_synthetic_data(df, num_rows)
                df = pd.concat([df, synthetic_data], ignore_index=True)
                X_res = df.drop(columns=[target_column])
                y_res = df[target_column]

            resampled_df = pd.DataFrame(X_res, columns=X.columns)
            resampled_df[target_column] = y_res

            st.write("Resampled Dataset:")
            st.write(resampled_df.head())

            st.write("Class distribution after resampling:")
            st.write(Counter(y_res))
            Imbalance.plot_class_distribution(y_res, "Resampled Dataset Class Distribution")

            csv = resampled_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download resampled dataset", data=csv, file_name='resampled_dataset.csv',
                               mime='text/csv')

    except Exception as e:
        st.error(f"An error occurred: {e}")


def main():

    # Display the gradient headline using HTML
    st.markdown("""
        <h1 style="font-family: 'Peace Sans', sans-serif; letter-spacing: normal;">
            <span style="background: -webkit-linear-gradient(left,#0000FF , #00FF00);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">Model Trainer and Predictor</span><br>
        </h1>
    """, unsafe_allow_html=True)

    # Upload dataset
    st.header('Upload Dataset')
    uploaded_file = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Uploaded dataset:')
        st.write(df.head())

        # Select target column
        target_column = st.selectbox('Select Target Column', df.columns)

        # Select model type
        model_type = st.radio('Select Model Type', ['Classification', 'Regression'])

        if st.button('Train Model'):
            train_models(df, target_column, model_type)

        # Provide new data for prediction
        st.header('Predict')
        new_data = {}
        for column in df.columns:
            if column != target_column:
                new_data[column] = st.text_input(f'Enter value for {column}', '')

        if st.button('Predict'):
            new_data_df = pd.DataFrame([new_data])
            predictions = load_and_predict_model(new_data_df)
            st.success(f'Predicted Output: {predictions}')

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
from txtfeature_engineering import preprocess_text, compute_bag_of_words, generate_word2vec_features, train_cbow, compute_tfidf, train_skipgram_model

# Streamlit app
def main():
    styled_title_ml = """
    <h1 style="font-family: 'Peace Sans', sans-serif; letter-spacing: normal;">
        <span style="background: -webkit-linear-gradient(left,  #0000FF , #00FF00);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;">TEXT FEATURE ENGINEERING
    </h1>
"""
    st.write(styled_title_ml, unsafe_allow_html=True)
    # File uploader for text or CSV files
    uploaded_file = st.file_uploader("Upload a text file or CSV file containing text data", type=["txt", "csv"])

    # Model selection
    model_types = ["Bag-of-Words", "Word2Vec", "CBOW", "TF-IDF", "Skipgram"]
    model_type = st.selectbox("Select model", model_types)

    if model_type == "TF-IDF":
        # Input for TF-IDF parameters
        ngram_min = st.number_input("Min N-gram", min_value=1, max_value=5, value=1)
        ngram_max = st.number_input("Max N-gram", min_value=1, max_value=5, value=2)
        top_n_tfidf = st.number_input("Top N TF-IDF results", min_value=1, max_value=50, value=10)

    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            # Process text file
            text_data = uploaded_file.read().decode("utf-8")
            sentences = [line for line in text_data.split('\n') if line.strip()]
            corpus = [preprocess_text(line) for line in sentences]  # Preprocess each line separately for TF-IDF
        elif uploaded_file.type == "text/csv":
            # Process CSV file
            df = pd.read_csv(uploaded_file)
            text_data = df.astype(str).apply(' '.join, axis=1)
            sentences = [text for text in text_data]
            corpus = [preprocess_text(text) for text in sentences]  # Preprocess each text entry for TF-IDF

        if st.button("Generate Features"):
            if sentences:
                if model_type == "Bag-of-Words":
                    st.subheader("Bag-of-Words Features")
                    bow_df = compute_bag_of_words(sentences)
                    st.write("Bag of Words:")
                    st.dataframe(bow_df)

                    # Download Button for Bag-of-Words CSV
                    csv = bow_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Bag of Words as CSV",
                        data=csv,
                        file_name='bag_of_words.csv',
                        mime='text/csv',
                    )

                elif model_type == "Word2Vec":
                    st.subheader("Word2Vec Features")
                    word2vec_features, word2vec_model = generate_word2vec_features(sentences)
                    st.write(word2vec_features)

                    # Download Button for Word2Vec CSV
                    csv = word2vec_features.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Word2Vec features as CSV",
                        data=csv,
                        file_name='word2vec_features.csv',
                        mime='text/csv',
                    )

                elif model_type == "CBOW":
                    st.subheader("CBOW Features")
                    tokenized_sentences = [preprocess_text(sentence) for sentence in sentences]
                    cbow_model = train_cbow(tokenized_sentences)
                    st.success("CBOW model trained successfully!")

                    words = cbow_model.wv.index_to_key[:10]  # Get the first 10 words
                    embeddings = {word: cbow_model.wv[word] for word in words}

                    # Download Button for CBOW embeddings CSV
                    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
                    csv = embeddings_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download CBOW embeddings as CSV",
                        data=csv,
                        file_name='cbow_embeddings.csv',
                        mime='text/csv',
                    )

                elif model_type == "Skipgram":
                    train_skipgram_model(sentences)

        # Compute TF-IDF for uploaded text/CSV data
        if model_type == "TF-IDF":
            with st.spinner("Computing TF-IDF..."):
                top_ngrams_tfidf = compute_tfidf(corpus, ngram_range=(ngram_min, ngram_max), top_n=top_n_tfidf)
            st.success("TF-IDF computation complete!")

            # Display top N-grams with their TF-IDF scores
            st.subheader(f"Top {top_n_tfidf} N-grams by TF-IDF score:")
            st.dataframe(top_ngrams_tfidf)

            # Download Button for TF-IDF results CSV
            csv_tfidf = top_ngrams_tfidf.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download TF-IDF results as CSV",
                data=csv_tfidf,
                file_name='top_ngrams_tfidf.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
