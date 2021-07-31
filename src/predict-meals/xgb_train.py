import argparse
import warnings

import neptune.new as neptune
import pandas as pd
from neptune.new.integrations.xgboost import NeptuneCallback
from xgboost import XGBRegressor

from data.dataset import load_dataset

warnings.filterwarnings("ignore")


def define_argparser():
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument("--token", type=str, default=None)
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

    run = neptune.init(project="ds-wook/predict-meals")
    neptune_callback = NeptuneCallback(run=run, log_tree=[0, 1, 2, 3])
    X_lunch = train[["월", "일", "요일", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]
    y_lunch = train["중식계"]
    X_test = test[["월", "일", "요일", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]

    lunch_params = pd.read_pickle("../../parameters/xgb_lunch_params.pkl")

    lunch_model = XGBRegressor(**lunch_params)
    lunch_model.fit(X_lunch, y_lunch, callbacks=[neptune_callback])

    lunch_preds = lunch_model.predict(X_test)

    X_dinner = train[["월", "일", "요일(석식)", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]
    y_dinner = train["석식계"]
    X_test = test[["월", "일", "요일(석식)", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]
    dinner_params = pd.read_pickle("../../parameters/xgb_dinner_params.pkl")
    dinner_model = XGBRegressor(**dinner_params)
    dinner_model.fit(X_dinner, y_dinner, callbacks=[neptune_callback])
    dinner_preds = dinner_model.predict(X_test)

    submission = pd.read_csv(path + "sample_submission.csv")
    submission["중식계"] = lunch_preds
    submission["석식계"] = dinner_preds

    submission.to_csv(args.submit + args.file, index=False)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
