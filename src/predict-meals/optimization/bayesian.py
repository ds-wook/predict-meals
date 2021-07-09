import warnings
from typing import Callable, Sequence, Union

import joblib
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


class BayesianOptimizer:
    def __init__(
        self, objective_function: Callable[[Trial], Union[float, Sequence[float]]]
    ):
        self.objective_function = objective_function

    def bulid_study(
        self, trials: FrozenTrial, name: str, liar: bool = False, verbose: bool = True
    ):
        if liar:
            sampler = TPESampler(
                seed=42,
                constant_liar=True,
                multivariate=True,
                group=True,
                n_startup_trials=20,
            )
            study = optuna.create_study(
                study_name=name, direction="minimize", sampler=sampler
            )
            study.optimize(self.objective_function, n_trials=trials)
        else:
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name=name, direction="minimize", sampler=sampler
            )
            study.optimize(self.objective_function, n_trials=trials)

        if verbose:
            self.display_study_statistics(study)

        return study

    @staticmethod
    def display_study_statistics(study: Study):
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    '{key}': {value},")

    @staticmethod
    def xgb_lunch_save_params(study: Study, params_name: str):
        params = study.best_trial.params
        params["random_state"] = 42
        params["n_estimators"] = 10000
        params["learning_rate"] = 0.02
        params["eval_metric"] = "mae"
        joblib.dump(params, "../../parameters/" + params_name)

    @staticmethod
    def xgb_dinner_save_params(study: Study, params_name: str):
        params = study.best_trial.params
        params["random_state"] = 42
        params["n_estimators"] = 10000
        params["eval_metric"] = "mae"
        joblib.dump(params, "../../parameters/" + params_name)


def xgb_lunch_objective(
    trial: FrozenTrial,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    verbose: Union[int, bool],
) -> float:
    param = {
        "lambda": trial.suggest_loguniform("lambda", 1e-03, 1e-01),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
        "random_state": 42,
        "learning_rate": 0.02,
        "n_estimators": 10000,
        "eval_metric": "mae",
    }
    model = XGBRegressor(**param)

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=verbose,
    )
    preds = model.predict(x_valid)

    mae = mean_absolute_error(y_valid, preds)

    return mae


def xgb_dinner_objective(
    trial: FrozenTrial,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    verbose: Union[int, bool],
) -> float:
    param = {
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "learning_rate": trial.suggest_float("learning_rate", 1e-02, 1e-01),
        "n_estimators": 10000,
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "random_state": 42,
        "eval_metric": "mae",
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
    }

    model = XGBRegressor(**param)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=verbose,
    )
    preds = model.predict(x_valid)

    mae = mean_absolute_error(y_valid, preds)

    return mae
