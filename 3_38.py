"""
決定木モデル
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from six import StringIO
import pydot


def main():
    iris = datasets.load_iris()
    # X = iris.data[:, [2, 3]]
    X = iris.data
    y = iris.target

    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state=0)
    clf.fit(X_train, y_train)

    # 評価指標の生成
    print("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
    print("Train - Confusion matrix :",metrics.confusion_matrix(y_train, clf.predict(X_train)))
    print("Train - classification report :", metrics.classification_report(y_train, clf.predict(X_train)))
    print("Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test)))
    print("Test - Confusion matrix :",metrics.confusion_matrix(y_test, clf.predict(X_test)))
    print("Test - classification report :", metrics.classification_report(y_test, clf.predict(X_test)))

    tree.export_graphviz(clf, out_file='outputs/tree.dot')

    out_data = StringIO()

    tree.export_graphviz(
        clf, 
        out_file=out_data,
        feature_names=iris.feature_names,
        class_names=clf.classes_.astype(int).astype(str),
        filled=True, 
        rounded=True,
        special_characters=True,
        node_ids=1,
    )
    graph = pydot.graph_from_dot_data(out_data.getvalue())
    #graph[0].write_png( 'outputs/iris.png' )
    #graph[0].write_pdf("outputs/iris.pdf")  # save to pdf


if __name__ == '__main__':
    main()