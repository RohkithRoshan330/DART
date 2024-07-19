# # Modules for mlpipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import PolynomialFeatures
# from xgboost import XGBClassifier
# import numpy as np
#
#
# class MLPIPE:
#     @staticmethod
#     # Function to perform classification using Logistic Regression
#     def logistic_regression(X_train, y_train, X_test, y_test):
#         lr_model = LinearRegression()
#         lr_model.fit(X_train, y_train)
#         y_pred = lr_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred.round())
#         precision = precision_score(y_test, y_pred.round())
#         recall = recall_score(y_test, y_pred.round())
#         f1 = f1_score(y_test, y_pred.round())
#         return accuracy, precision, recall, f1
#
#     @staticmethod
#     # Function to perform classification using Ridge Regression
#     def ridge_regression(X_train, y_train, X_test, y_test):
#         ridge_model = Ridge()
#         ridge_model.fit(X_train, y_train)
#         y_pred = ridge_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred.round())
#         precision = precision_score(y_test, y_pred.round())
#         recall = recall_score(y_test, y_pred.round())
#         f1 = f1_score(y_test, y_pred.round())
#         return accuracy, precision, recall, f1
#
#     @staticmethod
#     # Function to perform classification using Lasso Regression
#     def lasso_regression(X_train, y_train, X_test, y_test):
#         lasso_model = Lasso()
#         lasso_model.fit(X_train, y_train)
#         y_pred = lasso_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred.round())
#         precision = precision_score(y_test, y_pred.round())
#         recall = recall_score(y_test, y_pred.round())
#         f1 = f1_score(y_test, y_pred.round())
#         return accuracy, precision, recall, f1
#
#     @staticmethod
#     # Function to perform classification using Random Forest
#     def random_forest(X_train, y_train, X_test, y_test):
#         rf_model = RandomForestClassifier()
#         rf_model.fit(X_train, y_train)
#         y_pred = rf_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         return accuracy, precision, recall, f1
#
#     @staticmethod
#     # Function to perform classification using Naive Bayes
#     def naive_bayes(X_train, y_train, X_test, y_test):
#         nb_model = GaussianNB()
#         nb_model.fit(X_train, y_train)
#         y_pred = nb_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         return accuracy, precision, recall, f1
#
#     @staticmethod
#     # Function to perform classification using Decision Tree
#     def decision_tree(X_train, y_train, X_test, y_test):
#         dt_model = DecisionTreeClassifier()
#         dt_model.fit(X_train, y_train)
#         y_pred = dt_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         return accuracy, precision, recall, f1
#
#     @staticmethod
#     # Function to perform regression using Linear Regression
#     def linear_regression(X_train, y_train, X_test, y_test):
#         lr_model = LinearRegression()
#         lr_model.fit(X_train, y_train)
#         y_pred = lr_model.predict(X_test)
#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_test, y_pred)
#         return mse, mae, rmse, r2
#
#     @staticmethod
#     # Function to perform classification using AdaBoost
#     def adaboost(X_train, y_train, X_test, y_test):
#         ada_model = AdaBoostClassifier()
#         ada_model.fit(X_train, y_train)
#         y_pred = ada_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         return accuracy, precision, recall, f1
#
#     @staticmethod
#     # Function to perform classification using Gradient Boosting
#     def gradient_boosting(X_train, y_train, X_test, y_test):
#         gb_model = GradientBoostingClassifier()
#         gb_model.fit(X_train, y_train)
#         y_pred = gb_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         return accuracy, precision, recall, f1
#
#     @staticmethod
#     # Function to perform classification using XGBoost
#     def xgboost(X_train, y_train, X_test, y_test):
#         xgb_model = XGBClassifier()
#         xgb_model.fit(X_train, y_train)
#         y_pred = xgb_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         return accuracy, precision, recall, f1
#
#     @staticmethod
#     # Function to perform regression using Polynomial Regression
#     def polynomial_regression(X_train, y_train, X_test, y_test):
#         poly = PolynomialFeatures(degree=2)
#         X_train_poly = poly.fit_transform(X_train)
#         X_test_poly = poly.transform(X_test)
#         lr_model = LinearRegression()
#         lr_model.fit(X_train_poly, y_train)
#         y_pred = lr_model.predict(X_test_poly)
#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_test, y_pred)
#         return mse, mae, rmse, r2
#


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
from collections import defaultdict


class MLPIPE:

    def train_models(self, df, target_column, model_type):
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

    def load_and_predict_model(self, new_data):
        # Load the trained model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        # Make predictions
        predictions = model.predict(new_data)
        return predictions


