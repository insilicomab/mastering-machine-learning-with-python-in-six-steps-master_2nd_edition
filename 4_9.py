"""
層化K-分割交差検証のためのROC曲線のプロット
"""
import numpy as np
from numpy import interp as np_interp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from itertools import cycle


def main():
    # データの読み込み
    df = pd.read_csv("data/Diabetes.csv")
    X = df.iloc[:,:8].values     # 独立変数
    y = df['class'].values     # 従属変数

    # データの正規化
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

    # 学習データセットとテストデータセットに分けてモデルを評価
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2017)

    # 決定木を構築
    # clf = tree.DecisionTreeClassifier(random_state=2017)
    clf = LogisticRegression(random_state=2017)
    clf = clf.fit(X_train, y_train)

    kfold = model_selection.StratifiedKFold(n_splits=5)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    i = 0

    for (train, test), color in zip(kfold.split(X, y), colors):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # ROC曲線の計算
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += np_interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
    
    mean_tpr /= kfold.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()