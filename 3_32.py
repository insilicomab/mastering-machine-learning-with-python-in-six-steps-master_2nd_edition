"""
過剰適合、正しい適合、過小適合
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# 2つの変数の間の散布図を作成
def draw_plot(pos, neg, x1, x2):
    plt.figure(figsize=(6, 6))
    plt.scatter(np.extract(pos, x1), np.extract(pos, x2), c='b', marker='s', label='pos')
    plt.scatter(np.extract(neg, x1), np.extract(neg, x2), c='r', marker='o', label='neg')   
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()


# 変数1と2をその高次多項式に対応させる
def map_features(variable_1, variable_2, order):
    assert order >= 1
    def iter():
        for i in range(1, order + 1):
            for j in range(i + 1):
                yield np.power(variable_1, i - j) * np.power(variable_2, j)
    return np.vstack(iter())


# 分類直線を描画するための関数
def draw_boundary(classifier, order):
    dim = np.linspace(-0.8, 1.1, 100)
    dx, dy = np.meshgrid(dim, dim)
    v = map_features(dx.flatten(), dy.flatten(), order)
    z = (np.dot(classifier.coef_, v) + classifier.intercept_).reshape(100, 100)
    plt.contour(dx, dy, z, levels=[0], colors=['r'])


def main(args):
    data = pd.read_csv('data/LR_NonLinear.csv')
    pos = data['class'] == 1
    neg = data['class'] == 0
    x1 = data['x1']
    x2 = data['x2']

    out = map_features(data['x1'], data['x2'], order=args.order_no)
    X = out.transpose()
    y = data['class']

    # データを訓練用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # # c = 0.01として適合させる
    clf = LogisticRegression(C=0.01).fit(X_train, y_train)
    print ('Train Accuracy for C=0.01: ', clf.score(X_train, y_train))
    print ('Test Accuracy for C=0.01: ', clf.score(X_test, y_test))
    draw_plot(
        pos=pos,
        neg=neg,
        x1=x1,
        x2=x2,
    )
    plt.title('Fitting with C=0.01')
    draw_boundary(clf, order=args.order_no)
    plt.legend()
    plt.show()

    # # c = 1として適合させる
    clf = LogisticRegression(C=1).fit(X_train, y_train)
    print ('Train Accuracy for C=1: ', clf.score(X_train, y_train))
    print ('Test Accuracy for C=1: ', clf.score(X_test, y_test))
    draw_plot(
        pos=pos,
        neg=neg,
        x1=x1,
        x2=x2,
    )
    plt.title('Fitting with C=1')
    draw_boundary(clf, order=args.order_no)
    plt.legend()    
    plt.show()

    # # c = 10000として適合させる
    clf = LogisticRegression(C=10000).fit(X_train, y_train)
    print ('Train Accuracy for C=10000: ', clf.score(X_train, y_train))
    print ('Test Accuracy for C=10000: ', clf.score(X_test, y_test))
    draw_plot(
        pos=pos,
        neg=neg,
        x1=x1,
        x2=x2,
    )
    plt.title('Fitting with C=10000')
    draw_boundary(clf, order=args.order_no)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--order_no', type=int, default=10)
    
    args = parser.parse_args()

    main(args)