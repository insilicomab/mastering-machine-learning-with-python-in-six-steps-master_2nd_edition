"""
サポートベクターマシン
"""
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    print('Class labels:', np.unique(y))

    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = SVC(kernel='linear', C=1.0, random_state=0)
    clf.fit(X_train, y_train)

    # 評価指標の生成
    print("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
    print("Train - Confusion matrix :",metrics.confusion_matrix(y_train, clf.predict(X_train)))
    print("Train - classification report :", metrics.classification_report(y_train, clf.predict(X_train)))
    print("Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test)))
    print("Test - Confusion matrix :", metrics.confusion_matrix(y_test, clf.predict(X_test)))
    print("Test - classification report :", metrics.classification_report(y_test, clf.predict(X_test)))


if __name__ == '__main__':
    main()