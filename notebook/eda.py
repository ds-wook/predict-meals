# %%
import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
path_gothic = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fontprop = fm.FontProperties(fname=path_gothic, size=20)

# %%
path = "../input/predict-meals/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

# %%
train.head()
# %%
train.info()
# %%
train.columns = [
    "date",  # 일자
    "weekday",  # 요일
    "employees",  # 본사정원수
    "dayoff",  # 본사휴가자수
    "bustrip",  # 본사출장자수
    "overtime",  # 본사외무령승건
    "remote",  # 재택근무자수
    "breakfast",  # 조식메뉴
    "lunch",  # 중식메뉴
    "dinner",  # 석식메뉴
    "target_lunch",  # 중식계
    "target_dinner",  # 석식계
]
train.head()
# %%
train["date"] = pd.to_datetime(train["date"])
train.info()

# %%
test.head()
# %%
test.columns = [
    "date",  # 일자
    "weekday",  # 요일
    "employees",  # 본사정원수
    "dayoff",  # 본사휴가자수
    "bustrip",  # 본사출장자수
    "overtime",  # 본사외무령승건
    "remote",  # 재택근무자수
    "breakfast",  # 조식메뉴
    "lunch",  # 중식메뉴
    "dinner",  # 석식메뉴
]
test.head()
# %%
test["date"] = pd.to_datetime(test["date"])
test.info()
# %%


def to_datetime(df, date):
    df["date"] = pd.to_datetime(df[date])
    df["dow"] = pd.to_datetime(df[date]).dt.weekday + 1


to_datetime(train, "date")
to_datetime(test, "date")


# %%
def make_lunch(df):
    lunch = []
    for day in range(len(df)):
        tmp = df.iloc[day, 8].split(" ")  # 공백으로 문자열 구분
        tmp = " ".join(tmp).split()  # 빈 원소 삭제

        search = "("  # 원산지 정보는 삭제
        for menu in tmp:
            if search in menu:
                tmp.remove(menu)

        lunch.append(tmp)
    return lunch


# %%

train_lunch = make_lunch(train)
test_lunch = make_lunch(test)

# %%


def make_lunch_cols(lunch, df):
    # lunch train data에 메뉴명별 칼럼 만들기 (밥, 국, 반찬1-3)
    bob = []
    gook = []
    banchan1 = []
    banchan2 = []
    banchan3 = []
    kimchi = []
    side = []
    for i, day_menu in enumerate(lunch):
        bob_tmp = day_menu[0]
        bob.append(bob_tmp)
        gook_tmp = day_menu[1]
        gook.append(gook_tmp)
        banchan1_tmp = day_menu[2]
        banchan1.append(banchan1_tmp)
        banchan2_tmp = day_menu[3]
        banchan2.append(banchan2_tmp)
        banchan3_tmp = day_menu[4]
        banchan3.append(banchan3_tmp)

        if i < 1067:
            kimchi_tmp = day_menu[-1]
            kimchi.append(kimchi_tmp)
            side_tmp = day_menu[-2]
            side.append(side_tmp)
        else:
            kimchi_tmp = day_menu[-2]
            kimchi.append(kimchi_tmp)
            side_tmp = day_menu[-1]
            side.append(side_tmp)
    df_ln = df[
        [
            "date",
            "weekday",
            "dow",
            "employees",
            "dayoff",
            "bustrip",
            "overtime",
            "remote",
            "lunch",
        ]
    ]

    df_ln["bob"] = bob
    df_ln["gook"] = gook
    df_ln["banchan1"] = banchan1
    df_ln["banchan2"] = banchan2
    df_ln["banchan3"] = banchan3
    df_ln["kimchi"] = kimchi
    df_ln["side"] = side
    df_ln.head()
    return df_ln


# %%
train_ln = make_lunch_cols(train_lunch, train)
train_ln["target_lunch"] = train["target_lunch"]
test_ln = make_lunch_cols(test_lunch, test)

# %%
train_ln.info()
# %%
test_ln.info()

# %%

train_ln["weekday"] = train_ln["weekday"].map({"월": 0, "화": 1, "수": 2, "목": 3, "금": 4})
test_ln["weekday"] = test_ln["weekday"].map({"월": 0, "화": 1, "수": 2, "목": 3, "금": 4})

# %%
