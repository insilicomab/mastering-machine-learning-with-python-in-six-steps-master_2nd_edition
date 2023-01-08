"""
アンサンブルモデル
"""
# pip install mlxtend --upgrade
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# 現在はsklearnではなくmlxtendの一部として利用可能
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split


def main():
    # 擬似乱数の生成
    np.random.seed(2017)

    # データの読み込み
    df = pd.read_csv("data/Diabetes.csv")
    X = df.iloc[:,:8]     # 独立変数
    y = df['class']       # 従属変数

    # 訓練データセットとテストデータセットに分けてモデルを評価
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2017)
    LR = LogisticRegression(random_state=2017, max_iter=200)
    RF = RandomForestClassifier(n_estimators = 100, random_state=2017)
    SVM = SVC(random_state=0, probability=True)
    KNC = KNeighborsClassifier()
    DTC = DecisionTreeClassifier()
    ABC = AdaBoostClassifier(n_estimators = 100)
    BC = BaggingClassifier(n_estimators = 100)
    GBC = GradientBoostingClassifier(n_estimators = 100)
    clfs = []

    print('5-fold cross validation:\n')
    for clf, label in zip(
        [LR, RF, SVM, KNC, DTC, ABC, BC, GBC],
        [
            'Logistic Regression',
            'Random Forest',
            'Support Vector Machine',
            'KNeighbors',
            'Decision Tree',
            'Ada Boost',
            'Bagging',
            'Gradient Boosting'
        ]):
        scores = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy') 
        print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        md = clf.fit(X, y)
        clfs.append(md)
        print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_test), y_test)))


if __name__ == '__main__':
    main()