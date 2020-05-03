"""
This file contains utility functions for preparing
the dataset to answer the questions.
"""

# -- Imports
import numpy as np
import os
import pandas as pd
import requests
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import ShuffleSplit
import zipfile


# -- Globals
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip"
FILE_NAME = 'dataset_Facebook.csv'
SEP = ';'
COLUMNS = [
    'Lifetime Post Consumers', 'Category', 'Type', 'Page total likes',
    'Post Month', 'Post Weekday', 'Post Hour', 'Paid'
]
CATEG_COLUMNS = ['Type', 'Category', 'Post Month', 'Post Weekday']
Y_COL = 'Lifetime Post Consumers'
HOUR_COL = 'Post Hour'
HOUR_X_COL = 'hour_x'
HOUR_Y_COL = 'hour_y'
TEST_SIZE = 0.25


# -- Functions
def convert_hour(df, hour_col, hour_x_col, hour_y_col):
    """
        Convert hour_col to x and y polar coordinates
        to handle the fact that hour 0 is close to hour 23.
        We can then treat hour as a numeric feature.
    """
    assert hour_col in df.columns

    df[hour_x_col] = np.sin(2 * np.pi * df[hour_col] / 24)
    df[hour_y_col] = np.cos(2 * np.pi * df[hour_col] / 24)
    del df[hour_col]
    return df


def transform(series, transformer=None):
    """
        Transform input series using a quantile transformation,
        with target output a normal distribution.
        If a transformer object is provided, it is used
        to encode the input series.
    """
    if not transformer:
        # create and fit
        transformer = QuantileTransformer(output_distribution='normal')
        transformer.fit(series.values.reshape(-1, 1))

    transformed_series = pd.Series(
        transformer.transform(series.values.reshape(-1, 1)).ravel(),
        index=series.index
    )
    return transformed_series, transformer


def inverse_transform(series, transformer):
    """
        Inverse transform the input series using the provided PowerTransformer
    """
    transformed_series = transformer.inverse_transform(
        series.reshape(-1, 1))
    return pd.Series(transformed_series.ravel())


def prepare_datasets(
        df,
        number_of_splits=1, seed=0,
        do_convert_hour=True,
        do_transform=True
):
    """
        Prepare and split datasets for training purposes
    """
    # make sure we don't run into issues when the list gets changed
    categ_columns = CATEG_COLUMNS.copy()

    if do_convert_hour:
        if HOUR_COL in df.columns:
            df = convert_hour(df, HOUR_COL, HOUR_X_COL, HOUR_Y_COL)
    else:
        # add the hour into categorical columns
        if HOUR_COL not in categ_columns:
            categ_columns.append(HOUR_COL)

    # The categorical variables are encoded using one-hot encoding
    if categ_columns:
        df = pd.get_dummies(df, columns=categ_columns)

    # fill the NaN in the Paid column with the most common value
    # and convert to int
    # TODO Use KNN imputation
    df['Paid'] = df['Paid'].fillna(df.Paid.mode().iloc[0])
    df['Paid'] = df['Paid'].astype(int)

    shuffle_split = ShuffleSplit(n_splits=number_of_splits,
                                 test_size=TEST_SIZE,
                                 random_state=seed)
    y = df[Y_COL]
    X = df.drop(columns=Y_COL)

    for train_indices, test_indices in shuffle_split.split(df):
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]

        if do_transform:
            X_train['Page total likes'], pt_x = transform(
                X_train['Page total likes'])
            X_test['Page total likes'], _ = transform(
                X_test['Page total likes'], pt_x)

            y_train, transformer_y = transform(y_train)
            y_test, _ = transform(y_test, transformer_y)
        else:
            transformer_y = None

        yield X_train, X_test, y_train, y_test, transformer_y


def load_data(usecols=None):
    """
        Downloads the UCI dataset and reads
        a list of columns into a dataframe.
    """
    files_to_remove = []
    try:
        # Download file only if it's not available
        cwd = os.getcwd()  # current working directory
        filepath = os.path.join(cwd, FILE_NAME)

        if not os.path.exists(filepath):
            # Download the zip file into `zip_path`
            compressed_file = requests.get(URL)
            zip_filename = URL.split('/')[-1]
            zip_path = os.path.join(cwd, zip_filename)
            files_to_remove.append(zip_path)

            with open(zip_path, 'wb') as f:
                f.write(compressed_file.content)

            # Extract the zip fle into its components
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cwd)
                # get the downloaded files list
                files = zip_ref.namelist()
                # add the path for all, ignoring the csv file. We will keep that
                files = [os.path.join(cwd, f) for f in files if f != FILE_NAME]
                files_to_remove.extend(files)

        # Now read the file to a pandas DataFrame
        df = pd.read_csv(filepath, sep=SEP, usecols=usecols)
        return df

    except Exception:
        print('Error fetching data, exiting...')
        raise

    finally:
        if files_to_remove:
            for f in files_to_remove:
                # easier To ask for forgiveness
                try:
                    os.remove(f)
                except OSError:
                    # Just display the error, nothing more to do
                    print(f'Unable to delete {f}')
