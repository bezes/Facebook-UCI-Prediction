"""
This module contains functions for modelling the
Lifetime post consumers from the UCI dataset
"""

# -- Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from preparation import prepare_datasets, load_data, inverse_transform

# -- Constants
REGRESSORS_LIST = (
    DecisionTreeRegressor,
    Ridge,
    KNeighborsRegressor,
    SVR,
    RandomForestRegressor,
    XGBRegressor
)
RANDOM_SEARCH_PARAMS = {
    # 'Ridge': {},
    # 'LinearRegression': {},
    # 'DecisionTreeRegressor': {},
    'SVR': {
        'C': [1.0, 10.0, 100.0, 1000.0],
        'epsilon': np.logspace(-1, -4, 20),
        'tol': np.logspace(-2, -4, 10),
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf'],
    },
    'KNeighborsRegressor': {
        'n_neighbors': np.arange(3, 10),
        'p': [1, 2],
        'leaf_size': [20, 30, 40]
    },
    'RandomForestRegressor': {
        'n_estimators': np.arange(50, 500, 50),
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None, 5, 10, 20, 30, 40],
        'min_samples_split': [2, 3, 4, 5],
        # 'min_samples_leaf': [1, 2, 3],
        # 'bootstrap': [True, False]
    },
    'XGBRegressor': {
        'n_estimators': np.arange(50, 400, 50),
        'learning_rate': [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': np.arange(3, 15),
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
    },
}


def tune_regressor(model, X, y, random_state,
                   k_fold=5, n_iter=50, n_jobs=-1,
                   verbose=1):
    """
        Tunes the `model` based on a preconfigured
        hyperparameter dictionary and randomized search
        and returns the tuned model, i.e. initiated with
        the optimal hyperparameters
     """
    # Load the random grid
    try:
        random_grid = RANDOM_SEARCH_PARAMS[model.__name__]
    except KeyError:
        random_grid = {}

    if verbose >= 1:
        print('\nTuning {}...'.format(model.__name__))

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    model_obj = model()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    random_cv = RandomizedSearchCV(
        estimator=model_obj,
        scoring='neg_mean_squared_error',
        param_distributions=random_grid,
        n_iter=n_iter,
        cv=k_fold,
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs
    )

    random_cv.fit(X, y)

    best_model = model(**random_cv.best_params_)
    if verbose >= 2:
        print('\nBest parameters: {}'.format(random_cv.best_params_))

    results = random_cv.cv_results_
    i = random_cv.best_index_
    results_dict = {
        k: results[k][i] for k in
        ['mean_test_score', 'std_test_score',
         'mean_score_time', 'std_score_time',
         'mean_fit_time', 'std_fit_time'
         ]
    }

    return best_model, results_dict


def calc_predictions(model, X):
    return model.predict(X)


def select_model(
        X_train, y_train, regressors_list=REGRESSORS_LIST,
        k_fold=3, n_iter=10, n_jobs=-1, verbose=1):
    """
        This function tunes all models in the `regressors_list`
        and then chooses the best tuned model.

        The selection criterion chosen for now is maximum test score,
        i.e. minimum RMSE, without taking into consideration
        other parameters like fit and predict times.
        This can be tailored to the model needs,
        e.g. having a prediction time threshold above which
        we discard the model.

        The function returns the best model
        as well as a pandas dataframe with statistics
        on scores, fit times and predict times.
     """

    tuned_regressors = {}
    best_results = dict()

    for regressor in regressors_list:
        tuned_regressor, results_dict = tune_regressor(
            regressor, X_train, y_train, random_state=0,
            k_fold=k_fold, n_iter=n_iter, n_jobs=n_jobs,
            verbose=verbose)

        tuned_regressors[type(tuned_regressor).__name__] = tuned_regressor
        best_results[type(tuned_regressor).__name__] = results_dict

    # Now we need to choose the best between the tuned regressors
    results_df = pd.DataFrame(best_results)

    # Selection Criterion
    selected_model_str = results_df.loc['mean_test_score'].idxmax()
    selected_model_rmse = - results_df.loc['mean_test_score'].max()
    selected_model = tuned_regressors[selected_model_str]

    if verbose >= 1:
        print('\nBest model: {}, with RMSE: {:.03f}'.format(
            selected_model_str, selected_model_rmse
        ))

    return selected_model, results_df

# def main():
#     df = load_data()
#     numbers_of_splits = 2
#     seed = 0
#     categorical_columns = ['Type', 'Category', 'Post Month', 'Post Weekday']
#
#     for X_train, X_test, y_train, y_test, pt_y in prepare_datasets(
#             df,
#             number_of_splits=numbers_of_splits,
#             seed=seed,
#             do_convert_hour=True,
#             do_power_transform=True,
#             categ_columns=categorical_columns,
#     ):
#         selected_model, results_df = select_model(
#             X_train, y_train, regressors_list=REGRESSORS_LIST,
#             k_fold=3, n_iter=10)
#
#         # Evaluate the model performance
#         # for both train and test set
#
#
#         plt.show()
#         pass
#
#
# if __name__ == '__main__':
#     main()
