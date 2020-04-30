"""
This module contains functions for modelling the
Lifetime post consumers from the UCI dataset
"""

# -- Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import KFold, RandomizedSearchCV,\
    cross_validate
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from uci.preparation import prepare_datasets, load_data, inverse_transform


# -- Constants
RANDOM_SEARCH_PARAMS = {
    # 'Ridge': {'alpha': 1.0},
    # 'LinearRegression': {},
    'SVR': {
        'C': loguniform(1e0, 1e4),
        'gamma': loguniform(1e-4, 0.9),
        'kernel': ['rbf'],
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

def tune_regressor(model, X, y, random_state, k_fold=5, n_iter=50):
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

    if not random_grid:
        n_iter = 1

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
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    random_cv.fit(X, y)
    best_model = model(**random_cv.best_params_)

    return best_model


def calc_predictions(model, X):
    return model.predict(X)


def choose_regressor(X, y, regressors_list, k_fold=5):
    """
        This function runs a `k_fold` cross validation
        for all models in the `regressors_list`, and
        chooses the best one.

        The selection criterion chosen for now is minimum RMSE,
        without taking into consideration other parameters
        like fit and predict times. This can be tailored to
        the algorithm needs, e.g. having a prediction time
        threshold above which we ignore the model.

        The function returns the best model
        as well as a pandas dataframe with statistics
        on RMSE, fit times and predict times.
     """

    results_dict = {
        'rmse_mean': {},
        'rmse_stdev': {},
        'score_time_mean': {},
        'score_time_stdev': {},
        'fit_time_mean': {},
        'fit_time_stdev': {}
    }

    model_dict = {}
    for model in regressors_list:
        cv_results = cross_validate(
            model, X, y, cv=k_fold,
            scoring='neg_mean_squared_error'
        )
        model_dict[type(model).__name__] = model

        results_dict['rmse_mean'][type(model).__name__] = - np.mean(
            cv_results['test_score'])
        results_dict['rmse_stdev'][type(model).__name__] = np.std(
            cv_results['test_score'])

        results_dict['score_time_mean'][type(model).__name__] = np.mean(
            cv_results['score_time'])
        results_dict['score_time_stdev'][type(model).__name__] = np.std(
            cv_results['score_time'])

        results_dict['fit_time_mean'][type(model).__name__] = np.mean(
            cv_results['fit_time'])
        results_dict['fit_time_stdev'][type(model).__name__] = np.std(
            cv_results['fit_time'])

    results_df = pd.DataFrame(results_dict)

    # Selection Criterion
    selected_model_str = results_df['rmse_mean'].idxmin()
    selected_model = model_dict[selected_model_str]

    return selected_model, results_df


def main():
    df = load_data()
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

        tuned_regressors = []

        for regressor in regressors_list:
            tuned_regressor = tune_regressor(
                regressor, X_train, y_train, random_state=0, n_iter=10)
            tuned_regressors.append(tuned_regressor)

        # Now we need to choose the best between the tuned regressors
        selected_model, results_df = choose_regressor(
            X_train, y_train, tuned_regressors)

        # Plot results
        do_plot = False
        if do_plot:
            num_cols = len(results_df.columns)
            fig, axes = plt.subplots(1, num_cols // 2, figsize=(18, 6))
            for i in range(0, num_cols, 2):
                results_df.iloc[:, i].plot.bar(
                    ax=axes[i // 2],
                    yerr=results_df.iloc[:, i + 1] * 1.96
                )
                axes[i // 2].set_title(results_df.columns[i][:-5])
            plt.show()

        # Evaluate the model performance
        # for both train and test set

        # fit
        selected_model.fit(X_train, y_train)

        # predict
        y_train_pred = selected_model.predict(X_train)
        y_test_pred = selected_model.predict(X_test)

        # and denormalise both pred and original y series
        y_train_pred = inverse_transform(y_train_pred, pt_y)
        y_test_pred = inverse_transform(y_test_pred, pt_y)
        y_train = inverse_transform(y_train.values, pt_y)
        y_test = inverse_transform(y_test.values, pt_y)

        train_rmse = mean_squared_error(y_train, y_train_pred)
        test_rmse = mean_squared_error(y_test, y_test_pred)
        print('Train RMSE = {:.01f}, Test RMSE = {:.01f}'.format(
            train_rmse, test_rmse))

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].scatter(y_train, y_train_pred)
        axes[1, 0].scatter(y_train, y_train_pred)
        axes[0, 0].set_title('Training Set')
        axes[0, 1].scatter(y_test, y_test_pred)
        axes[1, 1].scatter(y_test, y_test_pred)
        axes[0, 1].set_title('Test Set')

        y = np.concatenate([y_train, y_test])
        ymax = np.max(y) * 1.01
        yperc95 = np.percentile(y, 95)
        axes[0, 0].set_xlim([0, ymax])
        axes[0, 0].set_ylim([0, ymax])

        axes[0, 1].set_xlim([0, ymax])
        axes[0, 1].set_ylim([0, ymax])

        axes[1, 0].set_xlim([0, yperc95])
        axes[1, 0].set_ylim([0, yperc95])

        axes[1, 1].set_xlim([0, yperc95])
        axes[1, 1].set_ylim([0, yperc95])

        for axx in axes:
            for ax in axx:
                # ax.set_ylim([0, ymax])
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.grid()
                ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--")

        plt.show()
        pass


if __name__ == '__main__':
    main()
