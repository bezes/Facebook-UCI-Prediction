"""
This module contains functions for modelling the
Lifetime post consumers from the UCI dataset
"""

# -- Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV,\
    cross_validate
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from uci.preparation import prepare_datasets, FILE_PATH, SEP, COLUMNS


# -- Constants
RANDOM_SEARCH_PARAMS = {
    'Lasso': {},
    'Ridge': {},
    # 'LinearRegression': {},
    'SVR': {
    },
    'RandomForestRegressor': {
        'n_estimators': np.linspace(100, 2000, 20).astype(int),
        'max_features': ['auto', 'sqrt'],
        'max_depth': np.linspace(5, 50, 10).astype(int),
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3],
        'bootstrap': [True, False]
    },
    'XGBRegressor': {
        'n_estimators': np.linspace(100, 2000, 20).astype(int),
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': np.arange(3, 15),
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
    }
}


# def evaluate(y_pred, y_test):
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     mape = 100 * np.mean(np.abs(y_pred - y_test) / y_test)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('RMSE: {:0.4f}'.format(rmse))
#     print('Accuracy = {:0.2f}%'.format(accuracy))
#     return rmse, accuracy


# def fit_lightgbm_model(X_train, y_train):
#     categ_cols = ['Type', 'Category', 'Post Month', 'Post Weekday']
#     assert set(categ_cols).issubset(set(X_train.columns))

def random_search(model, X, y, random_state, k_fold=5, n_iter=50):
    # Load the random grid
    try:
        random_grid = RANDOM_SEARCH_PARAMS[model.__name__]
    except KeyError:
        random_grid = {}

    if not random_grid:
        n_iter = 1

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    model_obj = model()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    random_cv = RandomizedSearchCV(
        estimator=model_obj,
        param_distributions=random_grid,
        n_iter=n_iter,
        cv=k_fold,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    random_cv.fit(X, y)
    best_model = model(**random_cv.best_params_)
    best_model.fit(X, y)

    return best_model


def calc_predictions(model, X):
    return model.predict(X)


def try_regressors(X, y, regressors_list, k_fold=5):
    """ Try regressors and find the best model """

    results_dict = {
        'rmse_mean': {},
        'rmse_stdev': {},
        'score_time_mean': {},
        'score_time_stdev': {},
        'fit_time_mean': {},
        'fit_time_stdev': {}
    }
    for model in regressors_list:
        cv_results = cross_validate(
            model, X, y, cv=k_fold,
            scoring=('r2', 'neg_mean_squared_error')
        )
        results_dict['rmse_mean'][type(model).__name__] = - np.mean(
            cv_results['test_neg_mean_squared_error'])
        results_dict['rmse_stdev'][type(model).__name__] = np.std(
            cv_results['test_neg_mean_squared_error'])

        results_dict['score_time_mean'][type(model).__name__] = np.mean(
            cv_results['score_time'])
        results_dict['score_time_stdev'][type(model).__name__] = np.std(
            cv_results['score_time'])

        results_dict['fit_time_mean'][type(model).__name__] = np.mean(
            cv_results['fit_time'])
        results_dict['fit_time_stdev'][type(model).__name__] = np.std(
            cv_results['fit_time'])

    results_df = pd.DataFrame(results_dict)
    return results_df


def main():
    df = pd.read_csv(FILE_PATH, sep=SEP, usecols=COLUMNS)
    numbers_of_sets = 1
    seed = 0
    dummy_columns = ['Type', 'Category', 'Post Month', 'Post Weekday']

    for X_train, X_test, y_train, y_test, pt_y in prepare_datasets(
            df,
            number_of_sets=numbers_of_sets,
            seed=seed,
            do_convert_hour=True,
            do_power_transform=True,
            dummy_columns=dummy_columns,
    ):
        regressors_list = [
            Lasso,
            Ridge,
            LinearRegression,
            SVR,
            RandomForestRegressor,
            XGBRegressor
        ]

        best_regressors = []

        for regressor in regressors_list:
            best_model = random_search(regressor, X_train,
                                       y_train, random_state=0,
                                       n_iter=20)
            best_regressors.append(best_model)

        results_df = try_regressors(
            X_train, y_train, best_regressors, k_fold=5)

        num_cols = len(results_df.columns)
        fig, axes = plt.subplots(1, num_cols // 2, figsize=(18, 6))
        for i in range(0, num_cols, 2):
            results_df.iloc[:, i].plot.bar(
                ax=axes[i // 2],
                yerr=results_df.iloc[:, i + 1] * 1.96
            )
            axes[i // 2].set_title(results_df.columns[i][:-5])
        plt.show()

        print(results_df.to_string())



if __name__ == '__main__':
    main()
