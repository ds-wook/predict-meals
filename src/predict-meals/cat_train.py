import argparse

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from data.dataset import load_dataset


def define_argparser():
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--submit", type=str, help="Input data save path", default="../../submissions/"
    )
    parse.add_argument("--path", type=str, default="../../input/predict-meals/")
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")

    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    path = args.path
    train, test = load_dataset(path)
    X_lunch = train[["월", "일", "요일", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]
    y_lunch = train["중식계"]
    x_train, x_valid, y_train, y_valid = train_test_split(
        X_lunch, y_lunch, test_size=0.15, random_state=42
    )
    cat_cols = cat_cols = ["월", "일", "요일", "본사시간외근무명령서승인건수"]
    train_data = Pool(data=x_train, label=y_train, cat_features=cat_cols)
    valid_data = Pool(data=x_valid, label=y_valid, cat_features=cat_cols)
    X_test = test[["월", "일", "요일", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]

    lunch_params = {
        "learning_rate": 0.03,
        "l2_leaf_reg": 0.4,
        "max_depth": 8,
        "bagging_temperature": 1,
        "min_data_in_leaf": 57,
        "max_bin": 494,
        "random_state": 42,
        "eval_metric": "MAE",
        "loss_function": "MAE",
        "iterations": 10000,
        "cat_features": cat_cols,
    }

    lunch_model = CatBoostRegressor(**lunch_params)
    lunch_model.fit(
        train_data,
        eval_set=valid_data,
        early_stopping_rounds=25,
        use_best_model=True,
        verbose=100,
    )

    lunch_preds = lunch_model.predict(X_test)

    X_dinner = train[["월", "일", "요일(석식)", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]
    y_dinner = train["석식계"]
    X_test = test[["월", "일", "요일(석식)", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]
    cat_cols = ["월", "일", "요일(석식)", "본사시간외근무명령서승인건수"]

    x_train, x_valid, y_train, y_valid = train_test_split(
        X_dinner, y_dinner, test_size=0.15, random_state=42
    )

    train_data = Pool(data=x_train, label=y_train, cat_features=cat_cols)
    valid_data = Pool(data=x_valid, label=y_valid, cat_features=cat_cols)
    dinner_params = {
        "learning_rate": 0.03,
        "l2_leaf_reg": 0.4,
        "max_depth": 8,
        "bagging_temperature": 1,
        "min_data_in_leaf": 57,
        "max_bin": 494,
        "random_state": 42,
        "eval_metric": "MAE",
        "loss_function": "MAE",
        "iterations": 10000,
        "cat_features": cat_cols,
    }

    dinner_model = CatBoostRegressor(**dinner_params)
    dinner_model.fit(
        train_data,
        eval_set=valid_data,
        early_stopping_rounds=25,
        use_best_model=True,
        verbose=100,
    )
    dinner_preds = dinner_model.predict(X_test)

    submission = pd.read_csv(path + "sample_submission.csv")
    submission["중식계"] = lunch_preds
    submission["석식계"] = dinner_preds

    submission.to_csv(args.submit + args.file, index=False)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
