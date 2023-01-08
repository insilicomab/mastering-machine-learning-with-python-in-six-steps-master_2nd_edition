"""
勾配ブースティング
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


def main():
    # データの読み込み
    df = pd.read_csv("Data/Diabetes.csv")

    # 弱い特徴量を使って決定木を作成する
    X = df[['age','serum_insulin']]     # 独立変数
    y = df['class'].values              # 従属変数

    # 正規化
    X = StandardScaler().fit_transform(X)

    # 学習データセットとテストデータセットに分けてモデルを評価する
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
    kfold = model_selection.StratifiedKFold(n_splits=5)
    num_trees = 100

    # Gradient Boostingを100回繰り返す
    clf_GBT = GradientBoostingClassifier(n_estimators=num_trees, learning_rate=0.1, random_state=2019).fit(X_train, y_train)
    results = model_selection.cross_val_score(clf_GBT, X_train, y_train, cv=kfold)

    print ("\nGradient Boosting - CV Train : %.2f" % results.mean())
    print ("Gradient Boosting - Train : %.2f" % metrics.accuracy_score(clf_GBT.predict(X_train), y_train))
    print ("Gradient Boosting - Test : %.2f" % metrics.accuracy_score(clf_GBT.predict(X_test), y_test))

    df= pd.read_csv('Data/digit.csv')
    X = df.iloc[:,1:17].values
    y = df['lettr'].values

    # 訓練データセットとテストデータセットに分けてモデルを評価
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
    kfold = model_selection.StratifiedKFold(n_splits=5)
    num_trees = 10
    clf_GBT = GradientBoostingClassifier(n_estimators=num_trees, learning_rate=0.1, random_state=2019).fit(X_train, y_train)
    results = model_selection.cross_val_score(clf_GBT, X_train, y_train, cv=kfold)
    print ("\nGradient Boosting - Train : %.2f" % metrics.accuracy_score
    (clf_GBT.predict(X_train), y_train))
    print ("Gradient Boosting - Test : %.2f" % metrics.accuracy_score
    (clf_GBT.predict(X_test), y_test))

    # 'T'という文字を予測して、予測精度がブースティングの反復ごとにどのように変化するかをみてみよう
    X_valid= (2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8)
    print ("Predicted letter: ", clf_GBT.predict([X_valid]))

    # 各段階ではブースティングの各反復で予測される確率を与える
    stage_preds = list(clf_GBT.staged_predict_proba([X_valid]))
    final_preds = clf_GBT.predict_proba([X_valid])

    # プロット
    x = range(1,27)
    label = np.unique(df['lettr'])
    plt.figure(figsize=(10,3))
    plt.subplot(131)
    plt.bar(x, stage_preds[0][0], align='center')
    plt.xticks(x, label)
    plt.xlabel('Label')
    plt.ylabel('Prediction Probability')
    plt.title('Round One')
    plt.autoscale()
    plt.subplot(132)
    plt.bar(x, stage_preds[5][0],align='center')
    plt.xticks(x, label)
    plt.xlabel('Label')
    plt.ylabel('Prediction Probability')
    plt.title('Round Five')
    plt.autoscale()
    plt.subplot(133)
    plt.bar(x, stage_preds[9][0],align='center')
    plt.xticks(x, label)
    plt.autoscale()
    plt.xlabel('Label')
    plt.ylabel('Prediction Probability')
    plt.title('Round Ten')
    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    main()