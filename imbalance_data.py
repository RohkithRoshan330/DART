import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class Imbalance:
    # Functions for Random Sampling
    def balance_dataset(X, y, method='under', sampling_strategy='auto', random_state=None):
        if method == 'under':
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'over':
            sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'SMOTE':
            smote = SMOTE()
            X_res, y_res = smote.fit_resample(X, y)
            return X_res, y_res
        elif method == 'ADASYN':
            adasyn = ADASYN()
            X_res, y_res = adasyn.fit_resample(X, y)
            return X_res, y_res
        elif method == 'Borderline-SMOTE':
            borderline_smote = BorderlineSMOTE()
            X_res, y_res = borderline_smote.fit_resample(X, y)
            return X_res, y_res
        else:
            raise ValueError("method should be 'under', 'over', 'SMOTE', 'ADASYN', or 'Borderline-SMOTE'")

        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res

    # Plotting functions
    def plot_class_distribution(y, title):
        counter = Counter(y)
        labels, values = zip(*counter.items())
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(title)
        st.pyplot(plt)

    # Synthetic Data Generation Function
    def generate_synthetic_data(df, num_rows):
        synthetic_data = pd.DataFrame(columns=df.columns)
        for _ in range(num_rows):
            synthetic_row = df.sample(n=1, replace=True)
            synthetic_data = pd.concat([synthetic_data, synthetic_row], ignore_index=True)
        return synthetic_data
