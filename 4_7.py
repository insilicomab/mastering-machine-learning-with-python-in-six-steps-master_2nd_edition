import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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

    # 10分割の交差検証でモデルを評価
    train_scores = cross_validate(clf, X_train, y_train, scoring='accuracy', cv=5)
    test_scores = cross_validate(clf, X_test, y_test, scoring='accuracy', cv=5)

    print ("Train Fold AUC Scores: ", train_scores['test_score'])
    print ("Train CV AUC Score: ", train_scores['test_score'].mean())
    print ("\nTest Fold AUC Scores: ", test_scores['test_score'])
    print ("Test CV AUC Score: ", test_scores['test_score'].mean())


if __name__ == '__main__':
    main()