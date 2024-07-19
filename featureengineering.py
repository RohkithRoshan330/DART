# Modules for featureengineering
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector, RFE, f_classif, mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


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

    @staticmethod
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

    @staticmethod
    def plot_feature_importances(X, y, feature_names, algorithm='Random Forest', n_estimators_rf=100,
                                 n_estimators_xgb=100, random_state=42):
        if algorithm == 'Random Forest':
            # Train a Random Forest classifier
            clf = RandomForestClassifier(n_estimators=n_estimators_rf, random_state=random_state)
        elif algorithm == 'XGBoost':
            # Train an XGBoost classifier
            clf = xgb.XGBClassifier(n_estimators=n_estimators_xgb, random_state=random_state)
        else:
            st.error("Invalid algorithm selected.")
            return

        # Fit the model
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


class FeatureSelection:
    def __init__(self, df):
        self.df = df

    # def filter_method(self, target_variable, num_features_to_select, method_selection):
    #     X = self.df.drop(target_variable, axis=1)
    #     y = self.df[target_variable]
    #     feature_scores = pd.DataFrame()

    #     if method_selection == "Fisher's Score":
    #         F, pval = f_classif(X, y)
    #         feature_scores = pd.DataFrame({'feature': X.columns, 'score': F})
    #     elif method_selection == "Information Gain":
    #         information_gain = mutual_info_classif(X, y)
    #         feature_scores = pd.DataFrame({'feature': X.columns, 'score': information_gain})
    #     elif method_selection == "Chi-Square Test":
    #         chi2_scores, p_values = chi2(X, y)
    #         feature_scores = pd.DataFrame({'feature': X.columns, 'score': chi2_scores})

    #     feature_scores = feature_scores.sort_values(by='score', ascending=False)

    #     if num_features_to_select > 0:
    #         selected_features = feature_scores.head(num_features_to_select)['feature'].tolist()
    #         selected_data = self.df[selected_features]
    #     else:
    #         selected_features = X.columns.tolist()
    #         selected_data = X

    #     return selected_features, feature_scores, selected_data

    # def warper_method(self, X, y, n_features, method):
    #     model = LinearRegression()
    #     if method == 'forward':
    #         sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward')
    #         sfs.fit(X, y)
    #         return sfs.get_support(), sfs.get_support(indices=True)
    #     elif method == 'backward':
    #         sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='backward')
    #         sfs.fit(X, y)
    #         return sfs.get_support(), sfs.get_support(indices=True)
    #     elif method == 'rfe':
    #         rfe = RFE(model, n_features_to_select=n_features)
    #         rfe.fit(X, y)
    #         return rfe.support_, rfe.ranking_
    #     else:
    #         raise ValueError("Method must be 'forward', 'backward', or 'rfe'.")

    # def embedding_method(self, target_variable, num_features_to_select):
    #     X = self.df.drop(target_variable, axis=1)
    #     y = self.df[target_variable]
    #     model = RandomForestClassifier(random_state=100)
    #     model.fit(X, y)
    #     importances = model.feature_importances_
    #     feature_scores = pd.DataFrame({'feature': X.columns, 'score': importances})
    #     feature_scores = feature_scores.sort_values(by='score', ascending=False)

    #     if num_features_to_select > 0:
    #         selected_features = feature_scores.head(num_features_to_select)['feature'].tolist()
    #         selected_data = self.df[selected_features]
    #     else:
    #         selected_features = X.columns.tolist()
    #         selected_data = X

    #     return selected_features, feature_scores, selected_data

    def forward_selection(X, y, n_features):
        model = LinearRegression()
        sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward')
        sfs.fit(X, y)
        return sfs.get_support(), sfs.get_support(indices=True)

    def backward_elimination(X, y, n_features):
        model = LinearRegression()
        sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='backward')
        sfs.fit(X, y)
        return sfs.get_support(), sfs.get_support(indices=True)

    def recursive_feature_elimination(X, y, n_features):
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=n_features)
        rfe.fit(X, y)
        return rfe.support_, rfe.ranking_

    # Function for filter feature selection
    def filter_method(df):
        st.header("Filter Method in Feature Selection")
        if st.checkbox("Show uploaded data (optional)"):
            st.dataframe(df)

        target_variable = st.selectbox("Select the target variable:", df.columns)
        num_features_to_select = st.number_input("Number of features to select (optional, enter 0 for all):",
                                                 min_value=1)

        selected_features = []
        feature_scores = pd.DataFrame()

        if df is not None:
            X = df.drop(target_variable, axis=1)
            y = df[target_variable]

            method_selection = st.selectbox("Select Feature Selection Method:",
                                            ("Fisher's Score", "Information Gain", "Chi-Square Test"))

            if method_selection == "Fisher's Score":
                F, pval = f_classif(X, y)
                feature_scores = pd.DataFrame({'feature': X.columns, 'score': F})
                feature_scores = feature_scores.sort_values(by='score', ascending=False)

            elif method_selection == "Information Gain":
                information_gain = mutual_info_classif(X, y)
                feature_scores = pd.DataFrame({'feature': X.columns, 'score': information_gain})
                feature_scores = feature_scores.sort_values(by='score', ascending=False)

            elif method_selection == "Chi-Square Test":
                chi2_scores, p_values = chi2(X, y)
                feature_scores = pd.DataFrame({'feature': X.columns, 'score': chi2_scores})
                feature_scores = feature_scores.sort_values(by='score', ascending=False)

            st.subheader("Extracted Features and Scores ")
            st.dataframe(feature_scores)

            # Plot the feature importances
            st.subheader("Feature Importances Graph")
            plt.figure(figsize=(10, 6))
            plt.xticks(rotation=45)
            sns.barplot(x="feature", y="score", data=feature_scores)
            st.pyplot(plt)

            if num_features_to_select > 0:
                selected_features = feature_scores.head(num_features_to_select)['feature'].tolist()
                selected_data = df[selected_features]
                selected_data = selected_data.reset_index(drop=True)
                st.subheader("Selected Features and Data")
                st.dataframe(selected_data)

        return selected_features, feature_scores if num_features_to_select > 0 else (X.columns, feature_scores)

    # Function for embedding feature selection
    def embedding_method(df):
        st.header("Embedding Method in Feature Selection")
        if st.checkbox("Show uploaded data (optional)"):
            st.dataframe(df)

        target_variable = st.selectbox("Select the target variable:", df.columns)
        num_features_to_select = st.number_input("Number of features to select (optional, enter 0 for all):",
                                                 min_value=1)

        selected_features = []
        feature_scores = pd.DataFrame()

        if df is not None:
            X = df.drop(target_variable, axis=1)
            y = df[target_variable]

            model = RandomForestClassifier(random_state=100)
            model.fit(X, y)
            importances = model.feature_importances_
            feature_scores = pd.DataFrame({'feature': X.columns, 'score': importances})
            feature_scores = feature_scores.sort_values(by='score', ascending=False)

            st.subheader("Extracted Features and Scores ")
            st.dataframe(feature_scores)

            # Plot the feature importances
            st.subheader("Feature Importances Graph")
            plt.figure(figsize=(10, 6))
            plt.xticks(rotation=45)
            sns.barplot(x="feature", y="score", data=feature_scores)
            st.pyplot(plt)

            if num_features_to_select > 0:
                selected_features = feature_scores.head(num_features_to_select)['feature'].tolist()
                selected_data = df[selected_features]
                selected_data = selected_data.reset_index(drop=True)
                st.subheader("Selected Features and Data ")
                st.dataframe(selected_data)

        return selected_features, feature_scores if num_features_to_select > 0 else (X.columns, feature_scores)