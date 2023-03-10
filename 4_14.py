import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn import metrics


def plot_decision_regions(X, y, classifier):
    h = .02  # メッシュサイズの設定
    # マーカーのカラーマップの設定
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # 決定境界のプロット設定
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl)

def main():
    # データの読み込み
    df = pd.read_csv("Data/Diabetes.csv")
    X = df.iloc[:,:8].values     # 独立変数
    y = df['class'].values     # 従属変数

    # 正規化
    X = StandardScaler().fit_transform(X)

    # PCA
    X = PCA(n_components=2).fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2017)

    kfold = model_selection.StratifiedKFold(n_splits=5)
    num_trees = 100

    seed = 2019 

    # 決定木
    clf_DT = DecisionTreeClassifier(random_state=seed).fit(X_train,y_train)
    results = model_selection.cross_val_score(clf_DT, X_train,y_train, cv=kfold)
    print("Decision Tree (stand alone) - Train : ", results.mean())
    print("Decision Tree (stand alone) - Test : ", metrics.accuracy_score(clf_DT.predict(X_test), y_test))

    # バギング
    clf_DT_Bag = BaggingClassifier(base_estimator=clf_DT, n_estimators=num_trees, random_state=seed).fit(X_train,y_train)
    results = model_selection.cross_val_score(clf_DT_Bag, X_train, y_train, cv=kfold)
    print("Decision Tree (Bagging) - Train : ", results.mean())
    print("Decision Tree (Bagging) - Test : ", metrics.accuracy_score(clf_DT_Bag.predict(X_test), y_test))

    # ランダムフォレスト
    clf_RF = RandomForestClassifier(n_estimators=num_trees).fit(X_train, y_train)
    results = model_selection.cross_val_score(clf_RF, X_train, y_train, cv=kfold)
    print("Random Forest - Train : ", results.mean())
    print("Random Forest  - Test : ", metrics.accuracy_score(clf_RF.predict(X_test), y_test))

    #エクストラツリー
    clf_ET = ExtraTreesClassifier(n_estimators=num_trees).fit(X_train, y_train)
    results = model_selection.cross_val_score(clf_ET, X_train, y_train, cv=kfold)
    print("ExtraTree - Train : ", results.mean())
    print("ExtraTree - Test : ", metrics.accuracy_score(clf_ET.predict(X_test), y_test))

    # 決定境界のプロット
    plt.figure(figsize=(10,6))
    plt.subplot(221)
    plot_decision_regions(X, y, clf_DT)
    plt.title('Decision Tree (Stand alone)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.subplot(222)
    plot_decision_regions(X, y, clf_DT_Bag)
    plt.title('Decision Tree (Bagging - 100 trees)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(loc='best')
    plt.subplot(223)
    plot_decision_regions(X, y, clf_RF)
    plt.title('RandomForest Tree (100 trees)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(loc='best')
    plt.subplot(224)
    plot_decision_regions(X, y, clf_ET)
    plt.title('Extream Random Tree (100 trees)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()