"""
正規化された残差とレバレッジのプロット
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_leverage_resid2


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

    # 機能名のリストを作成
    independent_variables = [
        'lotsize', 'bathrms','driveway', 'fullbase','gashw', 
        'airco','garagepl', 'prefarea','stories_one','stories_three'
    ]

    #リストを使って元のDataFrameからサブセットを選択
    X = df[independent_variables]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

    # フィッティングしたモデルを作成
    lm = sm.OLS(y_train, X_train).fit()

    fig, ax = plt.subplots(figsize=(8,6))
    fig = plot_leverage_resid2(lm, ax = ax)
    plt.show()


if __name__ == '__main__':
    main()