"""
決定木とAdaBoost
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


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

    # 5倍のクロスバリデーションを用いた決定木
    # より多くの不純物を含む葉を得るためにmax_depthを1に制限する
    clf_DT = DecisionTreeClassifier(max_depth=1, random_state=2019).fit(X_train,y_train)
    results = model_selection.cross_val_score(clf_DT, X_train,y_train, cv=kfold.split(X_train, y_train))

    print("Decision Tree (stand alone) - CV Train : %.2f" % results.mean())
    print("Decision Tree (stand alone) - Test : %.2f" % metrics.accuracy_score(clf_DT.predict(X_train), y_train))
    print("Decision Tree (stand alone) - Test : %.2f" % metrics.accuracy_score(clf_DT.predict(X_test), y_test))

    # Adaptive Boostingを100回繰り返す
    clf_DT_Boost = AdaBoostClassifier(base_estimator=clf_DT, n_estimators=num_trees, learning_rate=0.1, random_state=2019).fit(X_train,y_train)
    results = model_selection.cross_val_score(clf_DT_Boost, X_train, y_train, cv=kfold.split(X_train, y_train))

    print("\nDecision Tree (AdaBoosting) - CV Train : %.2f" % results.mean())
    print("Decision Tree (AdaBoosting) - Train : %.2f" % metrics.accuracy_score(clf_DT_Boost.predict(X_train), y_train))
    print("Decision Tree (AdaBoosting) - Test : %.2f" % metrics.accuracy_score(clf_DT_Boost.predict(X_test), y_test))


if __name__ == '__main__':
    main()