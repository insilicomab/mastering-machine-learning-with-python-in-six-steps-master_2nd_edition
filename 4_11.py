"""
決定木における特徴量の重要度
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def main():
    # データの読み込み
    df = pd.read_csv("Data/Diabetes.csv")
    X = df.iloc[:,:8].values     # 独立変数
    y = df['class'].values     # 従属変数

    # 正規化
    X = StandardScaler().fit_transform(X)

    # 学習データセットとテストデータセットに分けてモデルを評価
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
    kfold = model_selection.StratifiedKFold(n_splits=5)
    num_trees = 100

    # 決定木，5回の畳み込みによるクロスバリデーション
    clf_DT = DecisionTreeClassifier(random_state=2019).fit(X_train,y_train)
    results = model_selection.cross_validate(clf_DT, X_train, y_train, cv=kfold)

    print ("Decision Tree (stand alone) - Train : ", results['test_score'].mean())
    print ("Decision Tree (stand alone) - Test : ", metrics.accuracy_score(clf_DT.predict(X_test), y_test))

    # バギングを用いて100個の決定木モデルを構築し，平均/多数決の予測を行う
    clf_DT_Bag = BaggingClassifier(base_estimator=clf_DT, n_estimators=num_trees, random_state=2019).fit(X_train,y_train)
    results = model_selection.cross_validate(clf_DT_Bag, X_train, y_train, cv=kfold)

    print ("\nDecision Tree (Bagging) - Train : ", results['test_score'].mean())
    print ("Decision Tree (Bagging) - Test : ", metrics.accuracy_score(clf_DT_Bag.predict(X_test), y_test))

    feature_importance = clf_DT.feature_importances_

    # 重要度の高いものから相対的に重要度を上げる
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, df.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


if __name__ == '__main__':
    main()