"""
多重共線性と分散拡大係数
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm


def main():

    # データの読み込み
    df = pd.read_csv('data/Housing_Modified.csv')

    # バイナリを数値化されたブール値に変換
    lb = preprocessing.LabelBinarizer()
    df.driveway = lb.fit_transform(df.driveway)
    df.recroom = lb.fit_transform(df.recroom)
    df.fullbase = lb.fit_transform(df.fullbase)
    df.gashw = lb.fit_transform(df.gashw)
    df.airco = lb.fit_transform(df.airco)
    df.prefarea = lb.fit_transform(df.prefarea)

    # ダミー変数の作成
    df_stories = pd.get_dummies(df['stories'], prefix='stories', drop_first=True)

    # ダミー変数をメインのデータフレームに結合する
    df = pd.concat([df, df_stories], axis=1)
    del df['stories']

    # statmodels graphicsパッケージのplot_corを使って相関行列をプロット
    # 相関行列の作成
    corr = df.corr()
    sm.graphics.plot_corr(corr, xnames=list(corr.columns))
    plt.show()


if __name__ == "__main__":
    main()