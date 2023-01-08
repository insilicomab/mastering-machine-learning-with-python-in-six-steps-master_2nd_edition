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

    # 最初の差分を取る
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)

    # モデルの構築
    model = sm.tsa.ARIMA(ts_log, order=(3,1,2))
    results_ARIMA = model.fit()
    
    ts_predict = results_ARIMA.predict()

    # 差分を補うための補正
    predictions_ARIMA_diff = pd.Series(ts_predict, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

    plt.title('ARIMA Prediction - order(3,1,2)')
    plt.plot(ts_log, label='Actual')
    plt.plot(predictions_ARIMA_log, 'r--', label='Predicted')
    plt.xlabel('Year-Month')
    plt.ylabel('Sales')
    plt.legend(loc='best')
    plt.show()

    print("AIC: ", results_ARIMA.aic)
    print("BIC: ", results_ARIMA.bic)
    print("Mean Absolute Error: ", mean_absolute_error(ts_log_diff.values, ts_predict.values))
    print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(ts_log_diff.values, ts_predict.values)))

    # 自己相関の確認
    print("Durbin-Watson statistic :", sm.stats.durbin_watson(results_ARIMA.resid.values))


if __name__ == '__main__':
    main()