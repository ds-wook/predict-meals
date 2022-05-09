from typing import Optional

import pandas as pd
from neptune.new import Run
from omegaconf import DictConfig, open_dict
from optuna.integration import XGBoostPruningCallback
from optuna.trial import FrozenTrial
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from tuning.base import BaseTuner


class XGBoostTuner(BaseTuner):
    def __init__(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        valid_x: pd.DataFrame,
        valid_y: pd.Series,
        config: DictConfig,
        run: Optional[Run] = None,
    ):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        super().__init__(config, run)

    def _objective(self, trial: FrozenTrial, config: DictConfig) -> float:
        """
        Objective function
        Args:
            trial: trial object
            config: config object
        Returns:
            metric score
        """
        # trial parameters
        params = {
            "max_depth": trial.suggest_int("max_depth", *config.search.max_depth),
            "gamma": trial.suggest_float("gamma", *config.search.gamma),
            "colsample_bytree": trial.suggest_loguniform(
                "colsample_bytree",
                *config.search.colsample_bytree,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *config.search.learning_rate
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *config.search.min_child_weight
            ),
            "subsample": trial.suggest_float("subsample", *config.search.subsample),
            "reg_alpha": trial.suggest_loguniform(
                "reg_alpha", *config.search.reg_alpha
            ),
            "reg_lambda": trial.suggest_loguniform(
                "reg_lambda", *config.search.reg_lambda
            ),
        }

        # config update
        with open_dict(config.model):
            config.model.params.update(params)

        pruning_callback = XGBoostPruningCallback(trial, "validation_1-mae")
        model = XGBRegressor(**params)

        model.fit(
            self.train_x,
            self.train_y,
            eval_set=[(self.train_x, self.train_y), (self.valid_x, self.valid_y)],
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            callbacks=[pruning_callback],  # pruning과정을 callback할 수 있음.
            verbose=self.config.model.verbose,
        )

        preds = model.predict(self.valid_x)
        score = mean_absolute_error(self.valid_y.to_numpy(), preds)

        return score
