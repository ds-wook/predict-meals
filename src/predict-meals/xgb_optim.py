import argparse
from functools import partial

from sklearn.model_selection import train_test_split

from data.dataset import load_dataset
from optimization.bayesian import (
    BayesianOptimizer,
    xgb_dinner_objective,
    xgb_lunch_objective,
)


def define_argparser():
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--trials", type=int, default=150)
    parse.add_argument("--params", type=str, default="params.pkl")
    parse.add_argument("--liar", default=True, action="store_false")
    parse.add_argument("--verbose", type=int, default=False)
    parse.add_argument("--time", type=str, default="lunch")
    parse.add_argument("--path", type=str, default="../../input/predict-meals/")
    parse.add_argument("--name", default="Optimization", type=str)
    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    train, test = load_dataset(args.path)

    if args.time == "lunch":
        X_lunch = train[["월", "일", "요일", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]
        y_lunch = train["중식계"]
        x_train, x_valid, y_train, y_valid = train_test_split(
            X_lunch, y_lunch, test_size=0.15, random_state=42
        )
        objective = partial(
            xgb_lunch_objective,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            verbose=args.verbose,
        )

        bayesian_optim = BayesianOptimizer(objective)

        study = bayesian_optim.bulid_study(
            trials=args.trials, name=args.name, liar=args.liar
        )
        bayesian_optim.xgb_lunch_save_params(study, args.params)

    elif args.time == "dinner":
        X_dinner = train[["월", "일", "요일(석식)", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]]
        y_dinner = train["석식계"]
        x_train, x_valid, y_train, y_valid = train_test_split(
            X_dinner, y_dinner, test_size=0.15, random_state=42
        )
        objective = partial(
            xgb_dinner_objective,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            verbose=args.verbose,
        )

        bayesian_optim = BayesianOptimizer(objective)

        study = bayesian_optim.bulid_study(
            trials=args.trials, name=args.name, liar=args.liar
        )

        bayesian_optim.xgb_dinner_save_params(study, args.params)

    else:
        raise ValueError


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
