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
    model = sm.tsa.ARIMA(ts_log, order=(3,0,2))
    results_ARIMA = model.fit()
    
    # 将来値の予測
    ts_predict = results_ARIMA.predict('1971-06-01', '1972-05-01')
    plt.title('ARIMA Future Value Prediction - order(3,1,2)')
    plt.plot(ts_log, label='Actual')
    plt.plot(ts_predict, 'r--', label='Predicted')
    plt.xlabel('Year-Month')
    plt.ylabel('Sales')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()