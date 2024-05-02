#
#
#
# import streamlit as st
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
#
# # Function to encode categorical values
# def encode_categorical(dataframe, columns):
#     encoded_dataframe = dataframe.copy()
#     label_encoder = LabelEncoder()
#     for column in columns:
#         encoded_dataframe[column] = label_encoder.fit_transform(encoded_dataframe[column])
#     return encoded_dataframe
#
# # Streamlit app
# def main():
#     st.title("Categorical Value Encoder")
#
#     # File upload section
#     uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file)
#         st.write("Original Data:")
#         st.write(data)
#
#         # Select columns to encode
#         columns_to_encode = st.multiselect("Select columns to encode", data.columns)
#
#         if st.button("Encode"):
#             encoded_data = encode_categorical(data, columns_to_encode)
#             st.write("Encoded Data:")
#             st.write(encoded_data)
#
# # Run the app
# if __name__ == "__main__":
#     main()
#



import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import base64

# Function to encode categorical values
def encode_categorical(dataframe, columns):
    encoded_dataframe = dataframe.copy()
    label_encoder = LabelEncoder()
    for column in columns:
        encoded_dataframe[column] = label_encoder.fit_transform(encoded_dataframe[column])
    return encoded_dataframe

# Function to download CSV file
def download_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
    return href

# Streamlit app
def main():
    st.title("Categorical Value Encoder")

    # File upload section
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Original Data:")
        st.write(data)

        # Select columns to encode
        columns_to_encode = st.multiselect("Select columns to encode", data.columns)

        if st.button("Encode"):
            encoded_data = encode_categorical(data, columns_to_encode)
            st.write("Encoded Data:")
            st.write(encoded_data)

            # Download button for encoded data
            st.markdown(download_csv(encoded_data, "encoded_data"), unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()

# Importing necessary libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Define function to perform ML pipeline
def perform_ml_pipeline(data, target_column):
    # Split the dataset into features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical target column into numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize the classifier
    classifier = RandomForestClassifier()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# Define Streamlit app
def main():
    st.title("ML Pipeline with Streamlit")

    # File upload section
    uploaded_file = st.file_uploader("Upload a file", type=['csv', 'xls', 'txt'])
    if uploaded_file is not None:
        if uploaded_file.type == "application/vnd.ms-excel":
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)

        # Select target column
        target_column = st.selectbox("Select the target column", options=data.columns)

        # Perform ML pipeline
        accuracy = perform_ml_pipeline(data, target_column)
        st.subheader("Accuracy")
        st.write(accuracy)


# Run the app
if __name__ == "__main__":
    main()

