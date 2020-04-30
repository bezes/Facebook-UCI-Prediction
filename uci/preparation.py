"""
This file contains utility functions for preparing
the dataset to answer the questions.
"""

# -- Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

# -- Globals
FILE_PATH = 'dataset_Facebook.csv'
SEP = ';'
COLUMNS = [
    'Lifetime Post Consumers', 'Category', 'Type', 'Page total likes',
    'Post Month', 'Post Weekday', 'Post Hour', 'Paid'
]
Y_COL = 'Lifetime Post Consumers'
HOUR_COL = 'Post Hour'
HOUR_X_COL = 'hour_x'
HOUR_Y_COL = 'hour_y'
TEST_SIZE = 0.2


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


def construct_sets(df, y_col=Y_COL, test_size=TEST_SIZE, random_state=0):
    """
        Creates the X and y sets and splits to
        train and test based on input seed
    """
    y = df[y_col]
    X = df.drop(columns='Lifetime Post Consumers')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def power_transform(series, power_transformer=None):
    """
        Transform input series using a power transformation.
        We use the default sklearn method 'yeo-johnson', as
        it is safer than the box-cox (negative values available).
        If a PowerTransformer object is not provided, a new object
        is created and returned for future use.
        If a PowerTransformer object is provided, it is used
        to encode the input series.
    """
    if not power_transformer:
        # create and fit
        power_transformer = PowerTransformer()
        power_transformer.fit(series.values.reshape(-1, 1))

    transformed_series = pd.Series(
        power_transformer.transform(series.values.reshape(-1, 1)).ravel(),
        index=series.index
    )
    return transformed_series, power_transformer


def inverse_transform(series, power_transformer):
    """
        Inverse transform the input series using the provided PowerTransformer
    """
    transformed_series = power_transformer.inverse_transform(series.reshape(-1, 1))
    return pd.Series(transformed_series.ravel())


def prepare_datasets(
        df,
        number_of_sets=1, seed=0,
        do_convert_hour=True,
        do_power_transform=True,
        dummy_columns=None,
):
    if do_convert_hour and (HOUR_COL in df.columns):
        df = convert_hour(df, HOUR_COL, HOUR_X_COL, HOUR_Y_COL)

    if dummy_columns:
        df = pd.get_dummies(df, columns=dummy_columns)

    # fill the NaN in the Paid column with the most common value
    # and convert to int
    df['Paid'] = df['Paid'].fillna(df.Paid.mode().iloc[0]).astype(int)

    np.random.seed(seed)
    # kfold = KFold(n_splits=number_of_sets)
    # for train_indices, test_indices in k_fold.split(X):
    for _ in range(number_of_sets):
        random_state = np.random.randint(low=0, high=np.iinfo(int).max)

        X_train, X_test, y_train, y_test = construct_sets(
            df, random_state=random_state)

        if do_power_transform:
            X_train['Page total likes'], pt_x = power_transform(
                X_train['Page total likes'])
            X_test['Page total likes'], _ = power_transform(
                X_test['Page total likes'], pt_x)

            y_train, pt_y = power_transform(y_train)
            y_test, _ = power_transform(y_test, pt_y)
        else:
            pt_y = None

        yield X_train, X_test, y_train, y_test, pt_y
