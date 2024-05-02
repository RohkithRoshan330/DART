import streamlit as st
from poz_pp import data_preprocessing
from poz_pp import NLP
import pandas as pd

# Define Streamlit app title

st.write(
    f"""
    <h1 style="color: #00FF00;">
        <span style="font-weight: bold;">D</span>ART
    </h1>
    """,
    unsafe_allow_html=True
)
st.write(
    f"""
    <h5 style="color: white;">
        Make your process more easier
    </h5>
    """,
    unsafe_allow_html=True
)

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a file")

# Navigation bar for selecting section
section = st.sidebar.radio("Navigation", ["Data Preprocessing", "NLP"])


def poz_handle_missing_values(df, method):
    pass


if section == "Data Preprocessing":
    st.sidebar.subheader("Data Preprocessing")
    preprocess_option = st.sidebar.selectbox("Select a preprocessing option:", ["Handle Missing Values", "Standardize", "Normalize", "Data Integration", "Data Aggregation", "Split Data", "Handle Outliers"])

    if uploaded_file is not None:
        # Read file based on its extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['csv', 'xls', 'xlsx', 'txt', 'docx', 'doc', 'json']:
            data = data_preprocessing.poz_read_file(uploaded_file)

            # Display the uploaded data
            st.write("Uploaded Data:")
            st.write(data)

            if preprocess_option == "Handle Missing Values":
                processed_data = data_preprocessing.poz_handle_missing_values(data)
                st.subheader("Processed Data:")
                st.write(processed_data)

            st.title("Missing Values Handling")

            # File upload section
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

            if uploaded_file is not None:
                st.write("Uploaded CSV file:")
                df = pd.read_csv(uploaded_file)
                st.write(df)

                # Select method for handling missing values
                method = st.selectbox("Select method for handling missing values:",
                                      ["Mean", "Median", "Mode", "Constant"])

                if st.button("Handle Missing Values"):
                    if method == "Mean":
                        df_filled = poz_handle_missing_values(df, method='mean')
                    elif method == "Median":
                        df_filled = poz_handle_missing_values(df, method='median')
                    elif method == "Mode":
                        df_filled = poz_handle_missing_values(df, method='mode')
                    elif method == "Constant":
                        df_filled = poz_handle_missing_values(df, method='constant')
                    else:
                        st.error("Invalid method selected.")

                    if not df_filled.empty:
                        st.write("DataFrame with missing values handled:")
                        st.write(df_filled)

                        # Download button for processed CSV file
                        csv_file = df_filled.to_csv(index=False)
                        st.download_button(label="Download Processed CSV", data=csv_file,
                                           file_name="processed_data.csv", mime="text/csv")
            elif preprocess_option == "Standardize":
                processed_data = data_preprocessing.poz_standardize(data)
                st.subheader("Processed Data:")
                st.write(processed_data)

            elif preprocess_option == "Normalize":
                processed_data = data_preprocessing.poz_normalize(data)
                st.subheader("Processed Data:")
                st.write(processed_data)

            elif preprocess_option == "Data Integration":
                integrated_data = data_preprocessing.poz_data_integration(data)
                st.subheader("Integrated Data:")
                st.write(integrated_data)

            elif preprocess_option == "Data Aggregation":
                # Define grouping columns and aggregation functions
                group_by_columns = st.sidebar.multiselect("Select columns to group by:", data.columns)
                aggregation_functions = {}
                for column in data.columns:
                    if column not in group_by_columns:
                        aggregation_functions[column] = st.sidebar.selectbox(f"Select aggregation function for {column}:", ["mean", "sum", "count"])

                if st.sidebar.button("Aggregate Data"):
                    aggregated_data = data_preprocessing.poz_data_aggregation(data, group_by_columns, aggregation_functions)
                    st.subheader("Aggregated Data:")
                    st.write(aggregated_data)

            elif preprocess_option == "Split Data":
                target_column = st.sidebar.selectbox("Select the target column:", data.columns)
                test_size = st.sidebar.slider("Select the test size ratio:", 0.1, 0.5, 0.2)
                random_state = st.sidebar.number_input("Select random state:", value=42)
                axis = st.sidebar.radio("Select axis to drop target column:", [0, 1])

                if st.sidebar.button("Split Data"):
                    X_train, X_test, y_train, y_test = data_preprocessing.poz_split_data(data, target_column, test_size, random_state, axis)
                    st.subheader("Split Data:")
                    st.write("X_train:", X_train)
                    st.write("X_test:", X_test)
                    st.write("y_train:", y_train)
                    st.write("y_test:", y_test)

            elif preprocess_option == "Handle Outliers":
                method = st.sidebar.selectbox("Select method for outlier detection:", ["z-score", "iqr"])
                threshold = st.sidebar.number_input("Enter threshold value:", value=3.0)
                action = st.sidebar.selectbox("Select action for handling outliers:", ["remove", "transform", "cap"])

                if st.sidebar.button("Handle Outliers"):
                    cleaned_data = data_preprocessing.poz_outliers(data, method, threshold, action)
                    st.subheader("Processed Data without Outliers:")
                    st.write(cleaned_data)

        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, TXT, DOCX, DOC, or JSON file.")

elif section == "NLP":
    st.sidebar.subheader("NLP Tasks")
    nlp_option = st.sidebar.selectbox("Select an NLP task:",
                                      ["Tokenize Text", "Lowercase Text", "Remove Punctuation", "Stemming",
                                       "Expand Contractions", "Remove Special Characters", "Remove Noisy Text",
                                       "Word Embedding", "Padding", "Truncation", "Text Normalization",
                                       "Named Entity Recognition", "Part-of-Speech Tagging"])

    if nlp_option in ["Tokenize Text", "Lowercase Text", "Remove Punctuation", "Stemming", "Expand Contractions",
                      "Remove Special Characters", "Remove Noisy Text", "Text Normalization"]:
        st.subheader(nlp_option)
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8")
            text = st.text_area("Or enter text here:", value=file_content)
        else:
            text = st.text_area("Enter text:")
    else:
        text = st.text_area("Enter text:")

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
                st.download_button(label="Download Lowercased Text", data=lowercase_text, file_name="lowercase_text.txt")

    elif nlp_option == "Remove Punctuation":
        if st.button("Remove Punctuation"):
            text_without_punctuation = NLP.poz_remove_punctuation(text)
            st.subheader("Text without Punctuation:")
            st.write(text_without_punctuation)
            # Download button for text without punctuation
            if text_without_punctuation:
                st.download_button(label="Download Text without Punctuation", data=text_without_punctuation, file_name="text_without_punctuation.txt")

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
            text_without_special_chars = NLP.poz_remove_special_characters(text)
            st.subheader("Text without Special Characters:")
            st.write(text_without_special_chars)
            # Download button for text without special characters
            if text_without_special_chars:
                st.download_button(label="Download Text without Special Characters", data=text_without_special_chars, file_name="text_without_special_chars.txt")

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
            word_embedding = NLP.poz_word_embed(text)
            st.subheader("Word Embedding:")
            st.write(word_embedding)
            # Download button for word embedding
            if word_embedding:
                st.download_button(label="Download Word Embedding", data=word_embedding, file_name="word_embedding.txt")

    elif nlp_option == "Padding":
        sequence = st.text_input("Enter sequence to pad:", value=text)
        max_length = st.number_input("Enter max sequence length:")
        if st.button("Pad Sequence"):
            padded_sequence = NLP.poz_padding(sequence, max_length)
            st.subheader("Padded Sequence:")
            st.write(padded_sequence)
            # Download button for padded sequence
            if padded_sequence:
                st.download_button(label="Download Padded Sequence", data=padded_sequence, file_name="padded_sequence.txt")

    elif nlp_option == "Truncation":
        sequence = st.text_input("Enter sequence to truncate:", value=text)
        max_length = st.number_input("Enter max sequence length:")
        if st.button("Truncate Sequence"):
            truncated_sequence = NLP.poz_truncation(sequence, max_length)
            st.subheader("Truncated Sequence:")
            st.write(truncated_sequence)
            # Download button for truncated sequence
            if truncated_sequence:
                st.download_button(label="Download Truncated Sequence", data=truncated_sequence, file_name="truncated_sequence.txt")

    elif nlp_option == "Text Normalization":
        if st.button("Normalize Text"):
            normalized_text = NLP.poz_nlp_normalize(text)
            st.subheader("Normalized Text:")
            st.write(normalized_text)
            # Download button for normalized text
            if normalized_text:
                st.download_button(label="Download Normalized Text", data=normalized_text, file_name="normalized_text.txt")

    elif nlp_option == "Named Entity Recognition":
        if st.button("Recognize Entities"):
            entities = NLP.poz_name_entity_recognition(text)
            st.subheader("Named Entities:")
            st.write(entities)
            # Download button for named entities
            if entities:
                st.download_button(label="Download Named Entities", data=entities, file_name="named_entities.txt")

    elif nlp_option == "Part-of-Speech Tagging":
        if st.button("Tag Parts of Speech"):
            pos_tags = NLP.poz_part_of_speech(text)
            st.subheader("Part-of-Speech Tags:")
            st.write(pos_tags)
            # Download button for part

