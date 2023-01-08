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
    xgtrain = xgb.DMatrix(X_train, label=y_train, missing=-999)
    xgtest = xgb.DMatrix(X_test, label=y_test, missing=-999)
    num_rounds = 100

    # xgboostのパラメータを設定
    param = {
        'max_depth': 3,  # 各木の最大の深さ
        'objective': 'binary:logistic'
    }
    clf_xgb_cv = xgb.cv(
        param, xgtrain, 
        num_rounds,
        stratified=True,
        nfold=5,
        early_stopping_rounds=20,
        seed=2017
    )
    print ("Optimal number of trees/estimators is %i" % clf_xgb_cv.shape[0])
    watchlist  = [(xgtest,'test'), (xgtrain,'train')]
    clf_xgb = xgb.train(param, xgtrain,clf_xgb_cv.shape[0], watchlist)

    # predict関数で確率を生成
    # 0.5のカットオフを使って確率をクラスラベルに変換
    y_train_pred = (clf_xgb.predict(xgtrain, ntree_limit=clf_xgb.best_iteration) > 0.5).astype(int)
    y_test_pred = (clf_xgb.predict(xgtest, ntree_limit=clf_xgb.best_iteration) > 0.5).astype(int)
    print ("XGB - Train : %.2f" % metrics.accuracy_score(y_train_pred, y_train))
    print ("XGB - Test : %.2f" % metrics.accuracy_score(y_test_pred, y_test))


if __name__ == '__main__':
    main()