# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
#
#
# # Function to train model and get accuracy
# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy
#
#
# # Streamlit app
# st.title('Synthetic Data Comparison using SMOTE and Borderline SMOTE')
#
# # Upload datasets
# st.header('Upload Datasets')
# original_file = st.file_uploader("Upload the original dataset", type=["csv"])
# smote_file = st.file_uploader("Upload the SMOTE dataset", type=["csv"])
# borderline_smote1_file = st.file_uploader("Upload the Borderline SMOTE1 dataset", type=["csv"])
#
# if original_file and smote_file and borderline_smote1_file:
#     original_data = pd.read_csv(original_file)
#     smote_data = pd.read_csv(smote_file)
#     borderline_smote1_data = pd.read_csv(borderline_smote1_file)
#
#     # Assume the last column is the target
#     X_orig = original_data.iloc[:, :-1]
#     y_orig = original_data.iloc[:, -1]
#
#     X_smote = smote_data.iloc[:, :-1]
#     y_smote = smote_data.iloc[:, -1]
#
#     X_bl_smote1 = borderline_smote1_data.iloc[:, :-1]
#     y_bl_smote1 = borderline_smote1_data.iloc[:, -1]
#
#     # Calculate accuracies
#     acc_orig = train_and_evaluate(X_orig, y_orig)
#     acc_smote = train_and_evaluate(X_smote, y_smote)
#     acc_bl_smote1 = train_and_evaluate(X_bl_smote1, y_bl_smote1)
#
#     st.header('Accuracy Comparison')
#
#     # Create a DataFrame for accuracies
#     accuracy_df = pd.DataFrame({
#         'Dataset': ['Original', 'SMOTE', 'Borderline SMOTE1'],
#         'Accuracy': [acc_orig, acc_smote, acc_bl_smote1]
#     })
#
#     # Display accuracy bar chart with different colors
#     fig = px.bar(
#         accuracy_df,
#         x='Dataset',
#         y='Accuracy',
#         title='Accuracy Comparison',
#         color='Dataset',
#         color_discrete_sequence=px.colors.qualitative.Safe
#     )
#     st.plotly_chart(fig)
# else:
#     st.write("Please upload all the required datasets.")


import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Function to train model and get accuracy
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Streamlit app
st.title('Synthetic Data Comparison using SMOTE and Borderline SMOTE')

# Upload datasets
st.header('Upload Datasets')
original_file = st.file_uploader("Upload the original dataset", type=["csv"])
smote_file = st.file_uploader("Upload the SMOTE dataset", type=["csv"])
borderline_smote1_file = st.file_uploader("Upload the Borderline SMOTE1 dataset", type=["csv"])

if original_file and smote_file and borderline_smote1_file:
    original_data = pd.read_csv(original_file)
    smote_data = pd.read_csv(smote_file)
    borderline_smote1_data = pd.read_csv(borderline_smote1_file)

    # Assume the last column is the target
    X_orig = original_data.iloc[:, :-1]
    y_orig = original_data.iloc[:, -1]

    X_smote = smote_data.iloc[:, :-1]
    y_smote = smote_data.iloc[:, -1]

    X_bl_smote1 = borderline_smote1_data.iloc[:, :-1]
    y_bl_smote1 = borderline_smote1_data.iloc[:, -1]

    # Calculate accuracies
    acc_orig = train_and_evaluate(X_orig, y_orig)
    acc_smote = train_and_evaluate(X_smote, y_smote)
    acc_bl_smote1 = train_and_evaluate(X_bl_smote1, y_bl_smote1)

    st.header('Accuracy Comparison')

    # Create a DataFrame for accuracies
    accuracy_df = pd.DataFrame({
        'Dataset': ['Original', 'SMOTE', 'Borderline SMOTE1'],
        'Accuracy': [acc_orig, acc_smote, acc_bl_smote1]
    })

    # Display accuracy bar chart with different colors
    fig = px.bar(
        accuracy_df,
        x='Dataset',
        y='Accuracy',
        title='Accuracy Comparison',
        color='Dataset',
        color_discrete_sequence=px.colors.qualitative.Safe,
        text='Accuracy'  # This will display the accuracy score on the bars
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig)

    # Display accuracy scores as text
    st.header('Accuracy Scores')
    for index, row in accuracy_df.iterrows():
        st.write(f"{row['Dataset']}: {row['Accuracy']:.2%}")
else:
    st.write("Please upload all the required datasets.")
