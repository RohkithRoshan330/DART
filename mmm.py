import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
from collections import defaultdict

# Function to train models with all algorithms
def train_models(df, target_column, model_type):
    models = defaultdict()
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'Classification':
        # Classification models
        classification_models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Support Vector Classifier': SVC(),
            'AdaBoost Classifier': AdaBoostClassifier(),
            'Gradient Boosting Classifier': GradientBoostingClassifier(),
            'XGBoost Classifier': XGBClassifier(),
            'Decision Tree Classifier': DecisionTreeClassifier()
        }
        # Train and evaluate all classification models
        st.subheader('Classification Model Accuracies')
        for name, model in classification_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            models[name] = accuracy
            st.write(f'{name}: {accuracy:.4f}')

    elif model_type == 'Regression':
        # Regression models
        regression_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Support Vector Regressor': SVR(),
            'AdaBoost Regressor': AdaBoostRegressor(),
            'Gradient Boosting Regressor': GradientBoostingRegressor(),
            'XGBoost Regressor': XGBRegressor(),
            'Decision Tree Regressor': DecisionTreeRegressor()
        }
        # Train and evaluate all regression models
        st.subheader('Regression Model Mean Squared Errors')
        for name, model in regression_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            models[name] = -mse  # Using negative MSE for sorting
            st.write(f'{name}: {mse:.4f}')

    # Select model with highest accuracy or lowest mse
    best_model_name = max(models, key=models.get)
    best_model = classification_models.get(best_model_name) or regression_models.get(best_model_name)

    # Train the best model on the entire dataset
    best_model.fit(X, y)

    # Save the trained model to a pickle file
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    st.success(f'Model trained with {best_model_name} and saved as model.pkl')

# Function to load the model and make predictions
def load_and_predict_model(new_data):
    # Load the trained model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Make predictions
    predictions = model.predict(new_data)
    return predictions

# Streamlit UI
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






































