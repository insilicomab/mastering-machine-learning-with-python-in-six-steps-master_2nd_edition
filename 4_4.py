import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def find_optimal_cutoff(target, predicted):
    """
    イベントレートに関連する分類モデルの最適な確率カットオフポイントを見つけるためのパラメータ
    ----------
    target: 行が観測値。従属データまたは目的データを持つ行列
    predicted : 予測されたデータを持つ行列
    返り値
    最適なカットオフ値を持つリスト
    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    
    return list(roc_t['threshold']) 



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

    # 最適な確率の閾値を見つける
    # Note: probs[:, 1] は正のラベルである確率を持つ
    threshold = find_optimal_cutoff(y_train, probs[:, 1])
    print ("Optimal Probability Threshold: ", threshold)

    # 予測確率に閾値を適用する
    y_pred_optimal = np.where(y_pred_prob >= threshold, 1, 0)

    # 通常のアプローチと最適なカットオフの精度を比較する
    print ("\nNormal - Accuracy: ", metrics.accuracy_score(y_train, y_pred))
    print ("Optimal Cutoff - Accuracy: ", metrics.accuracy_score(y_train, y_pred_optimal))
    print ("\nNormal - Confusion Matrix: \n", metrics.confusion_matrix(y_train, y_pred))
    print ("Optimal - Cutoff Confusion Matrix: \n", metrics.confusion_matrix(y_train, y_pred_optimal))


if __name__ == '__main__':
    main()