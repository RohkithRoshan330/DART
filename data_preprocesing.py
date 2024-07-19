# Modules for data_preprocessing
import io
import os
import docx
import json
import numpy as np
import pandas as pd
from scipy.special import boxcox
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split

# Data_Preprocessing
class data_preprocessing:
    # To Read Files
    def poz_read_file(file_path_or_uploaded_file):
        # Check if the input is a file path or an uploaded file object
        if isinstance(file_path_or_uploaded_file, str):
            # It's a file path
            file_extension = os.path.splitext(file_path_or_uploaded_file)[-1].lower()
            with open(file_path_or_uploaded_file, 'rb') as f:
                file_content = f.read()
        else:
            # It's an uploaded file object
            file_extension = file_path_or_uploaded_file.name.split('.')[-1].lower()
            file_content = file_path_or_uploaded_file.getvalue()

        # Read the file content based on its extension
        if file_extension == 'csv':
            return pd.read_csv(io.BytesIO(file_content))
        elif file_extension in ['xls', 'xlsx']:
            return pd.read_excel(io.BytesIO(file_content))
        elif file_extension == 'txt':
            data = file_content.decode("utf-8")
            return pd.DataFrame({'text_data': [data]})
        elif file_extension in ['doc', 'docx']:
            doc = docx.Document(io.BytesIO(file_content))
            text = ' '.join([para.text for para in doc.paragraphs])
            return pd.DataFrame({'text_data': [text]})
        elif file_extension == 'json':
            json_data = json.loads(file_content)
            text = ' '.join(str(value) for value in json_data.values())
            return pd.DataFrame({'text_data': [text]})
        else:
            print("Unsupported file type.")
            return None

    # Handleing Missing Values (mean , median , mode , constant)
    def poz_handle_missing_values(df, fill_method='mean', constant=0):
        try:
            if fill_method == 'mean':
                fill_value = df.mean()
            elif fill_method == 'median':
                fill_value = df.median()
            elif fill_method == 'mode':
                fill_value = df.mode().iloc[0]
            elif fill_method == 'constant':
                fill_value = constant
            else:
                raise ValueError("Invalid fill_method. Options: 'mean', 'median', 'mode', 'constant'.")
            df_filled = df.fillna(fill_value)
        except Exception as e:
            print(f"An error occurred: {e}")
            df_filled = pd.DataFrame()  # Return an empty DataFrame in case of error
        return df_filled

    # Handleing Missing Values - Linear Regression
    def fill_missing_values_linear(df):
        df_filled = df.copy()
        initial_imputer = SimpleImputer(strategy='mean')
        df_initial_imputed = pd.DataFrame(initial_imputer.fit_transform(df_filled), columns=df.columns)
        for column in df.columns:
            missing_values_mask = df[column].isnull()
            if not missing_values_mask.any():
                continue
            X_train = df_initial_imputed.loc[~missing_values_mask].drop(columns=[column])
            y_train = df_initial_imputed.loc[~missing_values_mask, column]
            X_test = df_initial_imputed.loc[missing_values_mask].drop(columns=[column])
            model = LinearRegression()
            model.fit(X_train, y_train)
            predicted_values = model.predict(X_test)
            df_filled.loc[missing_values_mask, column] = predicted_values
        return df_filled

    #  Handleing Missing Values -  Random Forest Regressor
    def fill_missing_values_rf(df):
        df_filled = df.copy()
        initial_imputer = SimpleImputer(strategy='mean')
        df_initial_imputed = pd.DataFrame(initial_imputer.fit_transform(df_filled), columns=df.columns)
        for column in df.columns:
            missing_values_mask = df[column].isnull()
            if not missing_values_mask.any():
                continue
            X_train = df_initial_imputed.loc[~missing_values_mask].drop(columns=[column])
            y_train = df_initial_imputed.loc[~missing_values_mask, column]
            X_test = df_initial_imputed.loc[missing_values_mask].drop(columns=[column])
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            predicted_values = model.predict(X_test)
            df_filled.loc[missing_values_mask, column] = predicted_values
        return df_filled

    # Transforme Data
    def poz_transformation(df, method='standardize'):

        try:
            # Select numerical columns
            numerical_columns = df.select_dtypes(include='number')

            if method == 'standardize':
                # Standardize numerical columns using z-score normalization
                transformed_df = (numerical_columns - numerical_columns.mean()) / numerical_columns.std()

            elif method == 'boxcox':
                # Apply Box-Cox transformation
                transformed_df = numerical_columns.apply(lambda x: pd.Series(boxcox(x + 1)[0], index=x.index))

            elif method == 'yeojohnson':
                # Apply Yeo-Johnson power transformation
                pt = PowerTransformer(method='yeo-johnson')
                transformed_data = pt.fit_transform(numerical_columns)
                transformed_df = pd.DataFrame(transformed_data, columns=numerical_columns.columns,
                                              index=numerical_columns.index)

            elif method == 'normalize':
                # Normalize numerical columns
                transformed_df = numerical_columns.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

            elif method == 'scaler':
                # Scale numerical columns using StandardScaler
                scaler = StandardScaler()
                transformed_data = scaler.fit_transform(numerical_columns)
                transformed_df = pd.DataFrame(transformed_data, columns=numerical_columns.columns,
                                              index=numerical_columns.index)

            elif method == 'minmax':
                # Scale numerical columns using MinMaxScaler
                scaler = MinMaxScaler()
                transformed_data = scaler.fit_transform(numerical_columns)
                transformed_df = pd.DataFrame(transformed_data, columns=numerical_columns.columns,
                                              index=numerical_columns.index)

            elif method == 'log2':
                # Apply logarithmic transformation
                transformed_df = np.log2(numerical_columns)

            else:
                raise ValueError(
                    "Invalid method. Options: 'standardize', 'boxcox', 'yeojohnson', 'normalize', 'scaler', 'minmax', 'log2'.")

            # Combine transformed numerical columns with non-numeric columns
            for column in df.columns:
                if column in numerical_columns.columns:
                    df[column] = transformed_df[column]
        except Exception as e:
            print(f"An error occurred: {e}")
            df = pd.DataFrame()  # Return an empty DataFrame in case of error
        return df

    # Data Aggregation
    def poz_data_aggregation(data, group_by_columns, aggregation_functions):

        try:
            aggregated_data = data.groupby(group_by_columns).agg(aggregation_functions).reset_index()
            return aggregated_data
        except Exception as e:
            print(f"An error occurred during data aggregation: {e}")
            return None

    # Split Data
    def poz_split_data(data, target_column, test_size, random_state, axis):

        if axis not in [0, 1]:
            raise ValueError("Axis must be 0 or 1.")

        X = data.drop(target_column, axis=axis)
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # data = [X_train, X_test, y_train, y_test]
        return [X_train, X_test, y_train, y_test]

    # Outliers
    def poz_outliers(data, method='z-score', threshold=3, action='remove'):
        cleaned_data = data.copy()
        if method == 'z-score':
            # Detect outliers using z-score
            z_scores = (cleaned_data - cleaned_data.mean()) / cleaned_data.std()
            outliers = abs(z_scores) > threshold

        elif method == 'iqr':
            # Detect outliers using interquartile range (IQR)
            q1 = cleaned_data.quantile(0.25)
            q3 = cleaned_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (cleaned_data < lower_bound) | (cleaned_data > upper_bound)

        else:
            raise ValueError("Invalid method. Options: 'z-score', 'iqr'")

        if action == 'remove':
            # Remove outliers
            cleaned_data = cleaned_data[~outliers.any(axis=1)]

        # elif action == 'transform':
        #     # Transform outliers to the mean value
        #     cleaned_data[outliers] = cleaned_data.mean()

        elif action == 'cap':
            # Cap outliers to a specified range
            cleaned_data[outliers] = cleaned_data.clip(lower=lower_bound, upper=upper_bound, axis=1)

        else:
            raise ValueError("Invalid action. Options: 'remove', 'transform', 'cap'")

        return cleaned_data

    # Already used functions
    """ NOTE:- This functions are used in side the functions already  """
    def poz_standardize(df):
        """
        Standardize numerical data in a DataFrame using z-score normalization.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing numerical data.

        Returns:
        pandas.DataFrame: The DataFrame with numerical data standardized.
        """
        try:
            # Select numerical columns
            numerical_columns = df.select_dtypes(include='number')

            # Standardize numerical columns using z-score normalization
            standardized_df = (numerical_columns - numerical_columns.mean()) / numerical_columns.std()

            # Combine standardized numerical columns with non-numeric columns
            for column in df.columns:
                if column in numerical_columns.columns:
                    df[column] = standardized_df[column]
        except Exception as e:
            print(f"An error occurred: {e}")
            df = pd.DataFrame()  # Return an empty DataFrame in case of error

        return df

        def poz_normalize(df):

            try:
                # Select numerical columns
                numerical_columns = df.select_dtypes(include='number')

                # Normalize numerical columns using min-max normalization
                normalized_df = (numerical_columns - numerical_columns.min()) / (
                        numerical_columns.max() - numerical_columns.min())

                # Combine normalized numerical columns with non-numeric columns
                for column in df.columns:
                    if column in numerical_columns.columns:
                        df[column] = normalized_df[column]
            except Exception as e:
                print(f"An error occurred: {e}")
                df = pd.DataFrame()  # Return an empty DataFrame in case of error

            return df

        def poz_apply_pca(data, n_components):
            try:
                # Instantiate PCA with desired number of components
                pca = PCA(n_components=n_components)
                # Fit PCA model to data and transform data to lower-dimensional space
                transformed_data = pca.fit_transform(data)
                return transformed_data
            except Exception as e:
                print(f"An error occurred during PCA: {e}")
                return None

        def poz_data_integration(datasets):
            try:
                integrated_data = np.concatenate(datasets, axis=1)
                return integrated_data
            except Exception as e:
                print(f"An error occurred during data integration: {e}")
                return None
