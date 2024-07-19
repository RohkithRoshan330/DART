import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import wasserstein_distance
from imblearn.over_sampling import SMOTE
import great_expectations as ge


# Function to load datasets
def load_data(file):
    return pd.read_csv(file)


# Function to display dataset info
def display_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


# Function to train a model and evaluate accuracy
def evaluate_model(original_df, synthetic_df, target_column):
    X_original = original_df.drop(columns=[target_column])
    y_original = original_df[target_column]
    X_synthetic = synthetic_df.drop(columns=[target_column])
    y_synthetic = synthetic_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    synthetic_train_accuracy = accuracy_score(y_train, model.predict(X_train))
    synthetic_test_accuracy = accuracy_score(y_test, y_pred)
    original_test_accuracy = accuracy_score(y_original, model.predict(X_original))

    return synthetic_train_accuracy, synthetic_test_accuracy, original_test_accuracy, model


# Function to plot feature distributions
def plot_feature_distributions(original_df, synthetic_df, features):
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.histplot(original_df[feature], color="blue", label="Original", kde=True, stat="density", linewidth=0)
        sns.histplot(synthetic_df[feature], color="orange", label="Synthetic", kde=True, stat="density", linewidth=0)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        st.pyplot(plt)
        plt.close()


# Function to calculate statistical metrics
def calculate_statistics(original_df, synthetic_df, features):
    stats = []
    for feature in features:
        orig_mean = original_df[feature].mean()
        synth_mean = synthetic_df[feature].mean()
        orig_std = original_df[feature].std()
        synth_std = synthetic_df[feature].std()
        wass_dist = wasserstein_distance(original_df[feature], synthetic_df[feature])
        stats.append({
            'Feature': feature,
            'Original Mean': orig_mean,
            'Synthetic Mean': synth_mean,
            'Original Std': orig_std,
            'Synthetic Std': synth_std,
            'Wasserstein Distance': wass_dist
        })
    return pd.DataFrame(stats)


# Function to apply SMOTE and regenerate synthetic dataset
def apply_smote(synthetic_df, target_column):
    smote = SMOTE(random_state=42)
    X = synthetic_df.drop(columns=[target_column])
    y = synthetic_df[target_column]
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_column])],
                     axis=1)


# Function to display result for data quality check
def display_result(var):
    result = var["result"]
    success = var["success"]
    expectation_type = var["expectation_config"]["expectation_type"]
    column = var["expectation_config"]["kwargs"]["column"]

    success_str = f"{success}</span>"
    result_str = f"""
    <div style='border: 2px solid black; padding: 10px;'>
        <p><strong>Expectation Type:</strong> {expectation_type}</p>
        <p><strong>Column:</strong> {column}</p>
        <p><strong>Success:</strong> {success_str}</p>
        <p><strong>Element Count:</strong> {result['element_count']}</p>
        <p><strong>Unexpected Count:</strong> {result['unexpected_count']}</p>
        <p><strong>Unexpected Percent:</strong> {result['unexpected_percent']}%</p>
    </div>
    """
    st.markdown(result_str, unsafe_allow_html=True)


# Streamlit UI
st.title("Data Analysis and Quality Checking")

process_option = st.sidebar.selectbox("Select Process",
                                      ['Synthetic vs. Original Data Analysis', 'Resampled Data Quality Checking'])

if process_option == 'Synthetic vs. Original Data Analysis':
    st.sidebar.header("Upload your datasets")
    original_file = st.sidebar.file_uploader("Upload original dataset (CSV)", type=["csv"])
    synthetic_file = st.sidebar.file_uploader("Upload synthetic dataset (CSV)", type=["csv"])

    if original_file is not None and synthetic_file is not None:
        original_df = load_data(original_file)
        synthetic_df = load_data(synthetic_file)

        st.header("Original Dataset Info")
        st.write(original_df.head())
        display_info(original_df)

        st.header("Synthetic Dataset Info")
        st.write(synthetic_df.head())
        display_info(synthetic_df)

        features = original_df.columns.tolist()
        target_column = st.sidebar.selectbox("Select target column", features)

        if target_column:
            default_features = [feat for feat in features if feat != target_column][:3]

            st.header("Model Evaluation")
            synthetic_train_accuracy, synthetic_test_accuracy, original_test_accuracy, model = evaluate_model(
                original_df, synthetic_df, target_column)

            st.write(f"Synthetic Train Accuracy: {synthetic_train_accuracy:.2f}")
            st.write(f"Synthetic Test Accuracy: {synthetic_test_accuracy:.2f}")
            st.write(f"Original Test Accuracy: {original_test_accuracy:.2f}")

            st.header("Feature Distributions")
            selected_features = st.multiselect("Select features to compare", features, default=default_features)
            if selected_features:
                plot_feature_distributions(original_df, synthetic_df, selected_features)

            st.header("Statistical Comparison")
            stats = calculate_statistics(original_df, synthetic_df, selected_features)
            st.write(stats)

            if stats['Synthetic Mean'].mean() < stats['Original Mean'].mean():
                st.subheader("Regenerating Synthetic Data using SMOTE")
                regenerated_synthetic_df = apply_smote(synthetic_df, target_column)

                st.header("Regenerated Synthetic Dataset Info")
                st.write(regenerated_synthetic_df.head())
                display_info(regenerated_synthetic_df)

                st.header("Model Evaluation with Regenerated Data")
                _, regenerated_test_accuracy, _, _ = evaluate_model(original_df, regenerated_synthetic_df,
                                                                    target_column)
                st.write(f"Regenerated Synthetic Test Accuracy: {regenerated_test_accuracy:.2f}")

                csv = regenerated_synthetic_df.to_csv(index=False)
                st.download_button(label="Download Regenerated Dataset", data=csv,
                                   file_name='regenerated_synthetic_data.csv', mime='text/csv')
    else:
        st.write("Please upload both datasets.")

elif process_option == 'Resampled Data Quality Checking':
    st.title("Resampled Data Quality Checking")
    uploaded_file = st.file_uploader("Choose a Resampled CSV file", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        df = ge.from_pandas(data)
        st.write("File uploaded successfully!")
        st.write("Preview of the file:")
        st.write(data.head())

        preprocess = st.selectbox("Select Data Preprocessing Option",
                                  ('Completeness', 'Uniqueness', 'Validity', 'Consistency', 'Integrity'))

        if preprocess == 'Completeness':
            column = st.selectbox("Select the column for Completeness:", df.columns)
            var = df.expect_column_values_to_not_be_null(
                column=column,
                meta={
                    "dimension": "Completeness"
                }
            )
            display_result(var)

        elif preprocess == 'Uniqueness':
            column = st.selectbox("Select the column for Uniqueness:", df.columns)
            var = df.expect_column_values_to_be_unique(
                column=column,
                meta={
                    "dimension": 'Uniqueness'
                }
            )
            display_result(var)

        elif preprocess == 'Validity':
            column = st.selectbox("Select the column for Validity:", df.columns)
            regex = st.text_input("Enter the regex pattern for validity check (e.g., '[0-9]+')", value='(0[1-100])')
            var = df.expect_column_values_to_match_regex(
                column=column,
                regex=regex,
                meta={
                    "dimension": "Validity"
                }
            )
            display_result(var)

        elif preprocess == 'Consistency':
            column = st.selectbox("Select the column for Consistency:", df.columns)
            min_value = df[column].min()
            max_value = df[column].max()
            var = df.expect_column_values_to_be_between(
                column=column,
                min_value=min_value,
                max_value=max_value,
                meta={
                    "dimension": 'Consistency'
                }
            )
            display_result(var)

        elif preprocess == 'Integrity':
            column = st.selectbox("Select the column for Integrity:", df.columns)
            value_set = data[column].tolist()
            var = df.expect_column_values_to_be_in_set(
                column=column,
                value_set=value_set,
                meta={
                    "dimension": 'Integrity'
                }
            )
            display_result(var)
