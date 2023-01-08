"""
時系列の分解
データソース: O.D. Anderson (1976), in file: data/anderson14,  X社の月次売上高 Jan '65 – May '71 C.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def main():
    df = pd.read_csv('data/TS.csv')
    ts = pd.Series(list(df['Sales']), index=pd.to_datetime(df['Month'],format='%Y-%m'))

    # 対数変換
    ts_log = np.log(ts)
    ts_log.dropna(inplace=True)

    s_test = adfuller(ts_log, autolag='AIC')
    print ("Log transform stationary check p value: ", s_test[1])

    # 最初の差分を取る
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    s_test = adfuller(ts_log_diff, autolag='AIC')
    print ("First order difference stationary check p value: ", s_test[1] )

    # 移動平均で線を滑らかにする
    moving_avg = ts_log.rolling(12).mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,3))
    ax1.set_title('First order difference')
    ax1.tick_params(axis='x', labelsize=7)
    ax1.tick_params(axis='y', labelsize=7)
    ax1.plot(ts_log_diff)
    ax2.plot(ts_log)
    ax2.set_title('Log vs Moving AVg')
    ax2.tick_params(axis='x', labelsize=7)
    ax2.tick_params(axis='y', labelsize=7)
    ax2.plot(moving_avg, color='red')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()