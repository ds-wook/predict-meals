from typing import Tuple

import numpy as np
import pandas as pd


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")

    drops = ["조식메뉴", "중식메뉴", "석식메뉴"]

    train = train.drop(drops, axis=1)
    test = test.drop(drops, axis=1)

    train["월"] = pd.DatetimeIndex(train["일자"]).month
    test["월"] = pd.DatetimeIndex(test["일자"]).month

    train["일"] = pd.DatetimeIndex(train["일자"]).day
    test["일"] = pd.DatetimeIndex(test["일자"]).day

    weekday = {"월": 1, "화": 2, "수": 3, "목": 4, "금": 5}

    train["요일"] = train["요일"].map(weekday)
    test["요일"] = test["요일"].map(weekday)

    train["식사가능자수"] = train["본사정원수"] - train["본사휴가자수"] - train["현본사소속재택근무자수"]
    test["식사가능자수"] = test["본사정원수"] - test["본사휴가자수"] - test["현본사소속재택근무자수"]

    train["중식참여율"] = train["중식계"] / train["식사가능자수"]
    train["석식참여율"] = train["석식계"] / train["식사가능자수"]

    features = ["월", "일", "요일", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]
    labels = ["중식계", "석식계", "중식참여율", "석식참여율"]

    train = train[features + labels]
    test = test[features]

    # 요일을 석식 rank에 맞춰 mapping한 요일(석식) 칼럼 만들기.

    weekday_rank4dinner = {
        1: 1,
        2: 2,
        3: 5,
        4: 3,
        5: 4,
    }

    train["요일(석식)"] = train["요일"].map(weekday_rank4dinner)
    test["요일(석식)"] = test["요일"].map(weekday_rank4dinner)

    return train, test
