# %%
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    "date",
    "weekday",
    "employees",
    "dayoff",
    "bustrip",
    "overtime",
    "remote",
    "breakfast",
    "lunch",
    "dinner",
    "target_lunch",
    "target_dinner",
]
train.head()
# %%
train["date"] = pd.to_datetime(train["date"])
train.info()

# %%
train.plot(x="date", y="target_lunch", figsize=(40, 5))
plt.show()

# %%
train.query("date >= 2017")
# %%
train.loc[train["dinner"] == "*"]
# %%
train.loc[train["date"] == "2019-08-28"]["dinner"].values
# %%
train["dinner"] = train["dinner"].str.strip()
train.loc[train["dinner"] == "*"]
# %%
train.loc[train["target_dinner"] == 0]
# %%
train["weekday"].unique()
# %%
train.loc[train["date"] == "2019-01-02"]
# %%
