
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist


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

    # K Meansクラスター
    model = KMeans(n_clusters=3, random_state=11)
    model.fit(X)
    print (model.labels_)

    K = range(1,10)
    KM = [KMeans(n_clusters=k).fit(X) for k in K]
    centroids = [k.cluster_centers_ for k in KM]

    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/X.shape[0] for d in dist]

    # 正方形の中の合計
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X)**2)/X.shape[0]
    bss = tss-wcss
    varExplained = bss/tss*100
    kIdx = 10-1

    ### プロット ###
    kIdx = 2

    # elbowカーブ
    # プロットのサイズを設定
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(K, avgWithinSS, 'b*-')
    plt.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')

    plt.subplot(1, 2, 2)
    plt.plot(K, varExplained, 'b*-')
    plt.plot(K[kIdx], varExplained[kIdx], marker='o', markersize=12,
        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title('Elbow for KMeans clustering')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()