"""
多重共線性の除去
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence


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
        'lotsize', 'bedrooms', 'bathrms','driveway', 'recroom',
        'fullbase','gashw','airco','garagepl', 'prefarea',
        'stories_one','stories_two','stories_three'
    ]

    # リストを使って元のDataFrameからサブセットを選択
    X = df[independent_variables]
    y = df['price']

    thresh = 10
    
    for i in np.arange(0, len(independent_variables)):
        vif = [variance_inflation_factor(X[independent_variables].values, ix) for ix in range(X[independent_variables].shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print("vif :", vif)
            print("dropping " + X[independent_variables].columns[maxloc] + " at index: " + str(maxloc))
            del independent_variables[maxloc]
        else:
            break
    print('Final variables:', independent_variables)


if __name__ == "__main__":
    main()