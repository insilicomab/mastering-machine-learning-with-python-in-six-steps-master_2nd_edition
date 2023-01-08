"""
時系列の分解
データソース: O.D. Anderson (1976), in file: data/anderson14,  X社の月次売上高 Jan '65 – May '71 C.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm


def main():
    df = pd.read_csv('data/TS.csv')
    ts = pd.Series(list(df['Sales']), index=pd.to_datetime(df['Month'],format='%Y-%m'))

    # 対数変換
    ts_log = np.log(ts)
    ts_log.dropna(inplace=True)

    # モデルの構築
    model = sm.tsa.ARIMA(ts_log, order=(2,0,2))
    results_ARIMA = model.fit()
    
    ts_predict = results_ARIMA.predict()

    # モデルの評価
    print("AIC: ", results_ARIMA.aic)
    print("BIC: ", results_ARIMA.bic)
    print("Mean Absolute Error: ", mean_absolute_error(ts_log.values, ts_predict.values))
    print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(ts_log.values, ts_predict.values)))

    # 自己相関の確認
    print("Durbin-Watson statistic :", sm.stats.durbin_watson(results_ARIMA.resid.values))


if __name__ == '__main__':
    main()