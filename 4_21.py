"""
スタッキングモデル
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    seed = 2019
    np.random.seed(seed)  # 乱数の初期化
    # データの読み込み
    df = pd.read_csv("Data/Diabetes.csv")
    X = df.iloc[:,0:8] # 独立変数
    y = df['class'].values     # 従属変数

    # 正規化
    X = StandardScaler().fit_transform(X)
    # 訓練データセットとテストデータセットに分けて評価
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    kfold = model_selection.StratifiedKFold(n_splits=5)
    num_trees = 10
    verbose = True # to print the progress
    clfs = [
        KNeighborsClassifier(),
        RandomForestClassifier(n_estimators=num_trees, random_state=seed),
        GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
    ]

    # ブレンド用の訓練データセットとテストデータセットの作成
    dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

    print('5-fold cross validation:\n')
    for i, clf in enumerate(clfs):
        scores = model_selection.cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
        print("##### Base Model %0.0f #####" % i)
        print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        clf.fit(X_train, y_train)   
        print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_train), y_train)))
        dataset_blend_train[:,i] = clf.predict_proba(X_train)[:, 1]
        dataset_blend_test[:,i] = clf.predict_proba(X_test)[:, 1]  
        print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_test), y_test)))

    print ("##### Meta Model #####")
    clf = LogisticRegression()
    scores = model_selection.cross_val_score(clf, dataset_blend_train, y_train, cv=kfold, scoring='accuracy')
    clf.fit(dataset_blend_train, y_train)

    print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_train), y_train)))
    print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_test), y_test)))


if __name__ == '__main__':
    main()