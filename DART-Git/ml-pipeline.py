import base64
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, MultiTaskElasticNetCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

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






# Function to perform machine learning pipeline for classification
def classification_pipeline(data, target_column, test_size, selected_algorithms):
    try:
        # Preprocess data
        X = data.drop(columns=[target_column])
        y = data[target_column]
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize classifiers
        classifiers = {
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(),
            "Logistic Regression": LogisticRegression(),
            "XGBoost": XGBClassifier()
        }

        results = {}

        # Iterate over selected algorithms
        for name in selected_algorithms:
            clf = classifiers[name]

            # Train classifier
            clf.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

        return results

    except Exception as e:
        return {"Error": str(e)}

# Function to perform machine learning pipeline for regression
def regression_pipeline(data, target_column, test_size, selected_algorithms):
    try:
        # Preprocess data
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize regressors
        regressors = {
            "Random Forest": RandomForestRegressor(),
            "Support Vector Machine": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Linear Regression": LinearRegression(),
            "Multi-label Linear Regression": MultiTaskElasticNetCV(),
            "XGBoost": XGBRegressor()
        }

        results = {}

        # Iterate over selected algorithms
        for name in selected_algorithms:
            if name == "Polynomial Regression":
                # Apply Polynomial Regression
                model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                # Train regressor
                reg = regressors[name]
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)

            # Calculate evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store results
            results[name] = {"Mean Squared Error": mse, "Mean Absolute Error": mae, "R-squared Score": r2}

        return results

    except Exception as e:
        return {"Error": str(e)}

# Page title
st.title("Model Evaluation")

# Upload CSV file
file = st.file_uploader("Upload dataset", type=["csv"])

# Sidebar inputs
if file is not None:
    data = pd.read_csv(file)
    st.write("Preview of the dataset:")
    st.write(data.head())

    target_column = st.sidebar.selectbox("Select target column", data.columns)
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, value=0.3)
    selected_task = st.sidebar.radio("Select task", ["Classification", "Regression"])

    if selected_task == "Classification":
        selected_algorithms = st.sidebar.multiselect("Select classification algorithms", ["Random Forest", "Support Vector Machine",
                                                                                          "K-Nearest Neighbors", "Naive Bayes",
                                                                                          "Decision Tree", "Logistic Regression",
                                                                                          "XGBoost"])

        # Button to start classification
        if st.sidebar.button("Start Classification"):
            st.write("Processing...")

            # Run classification pipeline
            results = classification_pipeline(data, target_column, test_size, selected_algorithms)

            # Display classification results
            st.write("### Classification Results:")
            for name, acc in results.items():
                st.write(f"- {name}: Accuracy = {acc:.2f}")

    elif selected_task == "Regression":
        selected_algorithms = st.sidebar.multiselect("Select regression algorithms", ["Random Forest", "Support Vector Machine",
                                                                                      "K-Nearest Neighbors", "Linear Regression",
                                                                                      "Multi-label Linear Regression", "Polynomial Regression",
                                                                                      "XGBoost"])

        # Button to start regression
        if st.sidebar.button("Start Regression"):
            st.write("Processing...")

            # Run regression pipeline
            results = regression_pipeline(data, target_column, test_size, selected_algorithms)

            # Display regression results
            st.write("### Regression Results:")
            for name, metrics in results.items():
                st.write(f"- {name}:")
                for metric, value in metrics.items():
                    st.write(f"  - {metric}: {value:.2f}")
else:
    st.write("Please upload a CSV file.")
