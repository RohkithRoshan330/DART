# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
#
# # Load the dataset
# data = pd.read_csv('your_data.csv')
#
# # Preprocess the data (e.g., handle missing values, convert categorical variables)
# # For simplicity, let's assume the data is clean
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)
#
# # Train a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions on the testing data
# y_pred = model.predict(X_test)
#
# # Visualize the predictions
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# st.pyplot(plt)
#
# if __name__ == '__main__':
#     st.title('Exploratory Data Analysis with Predictions')
#     st.write('This app demonstrates exploratory data analysis with predictions using scikit-learn.')
#     st.write('Please select the dataset and model to use.')
#     # Add your dataset and model selection code here
#     st.pyplot(plt)

#
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
#
# # Load the dataset
# @st.cache
# def load_data():
#     data = pd.read_csv('your_data.csv')
#     return data
#
# data = load_data()
#
# # Preprocess the data (e.g., handle missing values, convert categorical variables)
# # For simplicity, let's assume the data is clean
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)
#
# # Train a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions on the testing data
# y_pred = model.predict(X_test)
#
# # Visualize the predictions
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
#
# # Display the plot
# st.pyplot(plt)
#
# if __name__ == '__main__':
#     st.title('Exploratory Data Analysis with Predictions')
#     st.write('This app demonstrates exploratory data analysis with predictions using scikit-learn.')
#     st.write('Please select the dataset and model to use.')
#     # Add your dataset and model selection code here
#     st.pyplot(plt)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Function to load data
def load_data(file):
    data = pd.read_csv(file)
    return data


# Train a linear regression model
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2,
                                                        random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


# Make predictions on the testing data
def predict(model, X_test):
    return model.predict(X_test)


# Visualize the predictions
def visualize_predictions(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    st.pyplot()


def main():
    st.title('Exploratory Data Analysis with Predictions')
    st.write('This app demonstrates exploratory data analysis with predictions using scikit-learn.')

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Load the data
        data = load_data(uploaded_file)
        st.write('**Dataset Preview:**')
        st.write(data.head())

        # Train the model
        model, X_test, y_test = train_model(data)

        # Make predictions
        y_pred = predict(model, X_test)

        # Visualize predictions
        st.write('**Visualization of Predictions:**')
        visualize_predictions(y_test, y_pred)


if __name__ == '__main__':
    main()
