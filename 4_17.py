"""
sklearnのラッパーを使ったxgboost
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


def main():
    # データの読み込み
    df = pd.read_csv("data/Diabetes.csv")
    predictors = ['age','serum_insulin']
    target = 'class'

    # 一般的な前処理としてラベルのエンコーディングと欠損値の処理を行う
    from sklearn import preprocessing
    for f in df.columns:
        if df[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))

    df.fillna((-999), inplace=True)

    # 決定木を構築するために弱い特徴量をいくつか使ってみよう
    X = df[['age','serum_insulin']] # 独立変数
    y = df['class'].values          # 従属変数

    # 標準化
    X = StandardScaler().fit_transform(X)

    # 弱い特徴量を使って木を作成する
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2017)
    num_rounds = 100
    kfold = model_selection.StratifiedKFold(n_splits=5)
    clf_XGB = XGBClassifier(n_estimators = num_rounds, objective= 'binary:logistic',seed=2017)

    # early_stopping_rounds を使用し，スコアが向上しなかった場合に cv を停止する
    clf_XGB.fit(X_train,y_train, early_stopping_rounds=20, eval_set=[(X_test, y_test)], verbose=False)
    results = model_selection.cross_val_score(clf_XGB, X_train,y_train, cv=kfold)

    print ("\nxgBoost - CV Train : %.2f" % results.mean())
    print ("xgBoost - Train : %.2f" % metrics.accuracy_score(clf_XGB.predict(X_train), y_train))
    print ("xgBoost - Test : %.2f" % metrics.accuracy_score(clf_XGB.predict(X_test), y_test))


if __name__ == '__main__':
    main()