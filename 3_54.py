"""
階層型クラスタリング
"""
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist


def main():
    iris = datasets.load_iris()

    # データフレームの変換
    iris = pd.DataFrame(
        data= np.c_[iris['data'], iris['target']],
        columns= iris['feature_names'] + ['species']
    )

    # カラム名からスペースを削除
    iris.columns = iris.columns.str.replace(' ','')
    iris.head()
    X = iris.iloc[:,:3]  # 独立変数
    y = iris.species   # 従属変数
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

    # 連結行列の生成
    Z = linkage(X, 'ward')
    c, coph_dists = cophenet(Z, pdist(X))
    # 系統樹の計算
    plt.figure(figsize=(25, 10))
    plt.title('Agglomerative Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # X軸のラベルを回転
        leaf_font_size=8.,  # X軸のラベルのフォントサイズを変更
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()