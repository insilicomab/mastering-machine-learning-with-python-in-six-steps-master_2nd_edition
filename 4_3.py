import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def main():
    # データの読み込み
    df = pd.read_csv("data/Diabetes.csv")
    print (df['class'].value_counts(normalize=True))

    X = df.iloc[:,:8]     # 独立変数
    y = df['class']     # 従属変数

    # 訓練データセットとテストデータセットに分けてモデルを評価
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # ロジスティック回帰モデルをインスタンス化して適合
    model = LogisticRegression(max_iter=150)
    model = model.fit(X_train, y_train)

    # 訓練セットのクラスラベルを予測する。predict関数は確率が0.5より大きい値を1 か 0に変換する
    y_pred = model.predict(X_train)

    # クラス確率の生成
    # probs配列には2つの要素が返されることに注意
    # 1番目の要素は負のクラスの確率
    # 2番目の要素は正のクラスの確率
    probs = model.predict_proba(X_train)
    y_pred_prob = probs[:, 1]

    # 評価指標の生成
    print ("Accuracy: ", metrics.accuracy_score(y_train, y_pred))

    # 偽陽性、真陽性率の抽出
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    i = np.arange(len(tpr)) # Dataframeのインデックス
    roc = pd.DataFrame({
        'fpr' : pd.Series(fpr, index=i),
        'tpr' : pd.Series(tpr, index = i),
        '1-fpr' : pd.Series(1-fpr, index = i), 
        'tf' : pd.Series(tpr - (1-fpr), index = i),
        'thresholds' : pd.Series(thresholds, index = i)
    })
    print(roc.head())

    # tprと1-fprプロットして比較
    fig, ax = plt.subplots()
    plt.plot(roc['tpr'], label='tpr')
    plt.plot(roc['1-fpr'], color = 'red', label='1-fpr')
    plt.legend(loc='best')
    plt.xlabel('1-False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()


if __name__ == '__main__':
    main()