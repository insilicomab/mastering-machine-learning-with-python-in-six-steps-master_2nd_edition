"""
時系列の分解
データソース: O.D. Anderson (1976), in file: data/anderson14,  X社の月次売上高 Jan '65 – May '71 C.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,3))

    # ACFチャート
    fig = sm.graphics.tsa.plot_acf(ts_log_diff.values.squeeze(), lags=20, ax=ax1)

    # 95%信頼区間線の描画
    ax1.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    ax1.set_xlabel('Lags')

    # PACFチャート
    fig = sm.graphics.tsa.plot_pacf(ts_log_diff, lags=20, ax=ax2)

    # 95%信頼区間線の描画
    ax2.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    ax2.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    ax2.set_xlabel('Lags')

    plt.show()


if __name__ == '__main__':
    main()