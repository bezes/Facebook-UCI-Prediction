"""
This module contains functions for modelling the
Lifetime post consumers from the UCI dataset
"""

# -- Imports
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from preparation import inverse_transform

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
    'SVR': {
        'C': [1.0, 10.0, 100.0],
        'epsilon': np.logspace(-1, -4, 10),
        'tol': np.logspace(-1, -4, 10),
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf'],
    },
    'KNeighborsRegressor': {
        'n_neighbors': np.arange(3, 10),
        'p': [1, 2],
    },
    'RandomForestRegressor': {
        'n_estimators': np.arange(10, 200, 10),
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None, 5, 10, 20, 30, 40],
        'min_samples_split': [2, 3, 4, 5],
    },
    'XGBRegressor': {
        'booster': ['gbtree'],
        'max_depth': np.arange(3, 10),
        'learning_rate': np.arange(0.01, 1.0, 0.01),
        'n_estimators': np.arange(10, 500, 10),
        'gamma': np.arange(0.0, 5.0, 0.1),
        'subsample': np.arange(0.5, 1.1, 0.1),
        'colsample_bytree': np.arange(0.5, 1.1, 0.1),
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
        print('### Tuning {}... ###'.format(model.__name__))

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
        print('### Best parameters: ###\n{}'.format(random_cv.best_params_))

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


def select_model(
        X_train, y_train, regressors_list=REGRESSORS_LIST,
        k_fold=3, n_iter=10, n_jobs=-1, verbose=1,
        random_state=0
):
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
            regressor, X_train, y_train, random_state=random_state,
            k_fold=k_fold, n_iter=n_iter, n_jobs=n_jobs,
            verbose=verbose)

        tuned_regressors[type(tuned_regressor).__name__] = tuned_regressor
        best_results[type(tuned_regressor).__name__] = results_dict

    # Now we need to choose the best between the tuned regressors
    results_df = pd.DataFrame(best_results)

    # Selection Criterion
    selected_model_str = results_df.loc['mean_test_score'].idxmax()
    selected_model = tuned_regressors[selected_model_str]

    if verbose >= 1:
        print('\n### Best model: {} ###'.format(selected_model_str))

    return selected_model, results_df


def get_predictions(model, X_train, y_train, X_test, y_test,
                    transformer_y=None):
    # Predict both train and test set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if transformer_y:
        # denormalise both pred and original y series
        y_train_pred = inverse_transform(y_train_pred, transformer_y)
        y_test_pred = inverse_transform(y_test_pred, transformer_y)
        y_train = inverse_transform(y_train.values, transformer_y)
        y_test = inverse_transform(y_test.values, transformer_y)

    return y_train_pred, y_test_pred, y_train, y_test


def calc_rmse(y_train_pred, y_test_pred, y_train, y_test):
    """
        Calculate root mean squared error for train and test predictions
    """
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return train_rmse, test_rmse


def calc_mae(y_train_pred, y_test_pred, y_train, y_test):
    """
        Calculate root mean squared error for train and test predictions
    """
    train_rmse = mean_absolute_error(y_train, y_train_pred)
    test_rmse = mean_absolute_error(y_test, y_test_pred)

    return train_rmse, test_rmse
