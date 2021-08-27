# predict-meals
구내식당 식수 인원 예측 AI 경진대회


## Benchmark
|Type|MAE|Time|
|----|---|----|
|Random|73.7870|**15.5 s**|
|Gaussian|73.9332|1min 31s|
|**TPE**|**72.2271**|21.9 s|

## Requirements
+ numpy
+ pandas
+ scikit-learn
+ xgboost
+ optuna
+ neptune
