
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm
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

    score = []
    for n_clusters in range(2,10):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        score.append(silhouette_score(X, labels, metric='euclidean'))

    # プロットのサイズを設定
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(score)
    plt.grid(True)
    plt.ylabel("Silouette Score")
    plt.xlabel("k")
    plt.title("Silouette for K-means")

    # n_clustersの値とランダム生成器でクラスターを初期化する
    model = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
    model.fit_predict(X)
    cluster_labels = np.unique(model.labels_)
    n_clusters = cluster_labels.shape[0]

    # 各サンプルのシルエット・スコアを計算する
    silhouette_vals = silhouette_samples(X, model.labels_)

    plt.subplot(1, 2, 2)

    # colormapのスペクトル値を取得
    cmap = cm.get_cmap("Spectral")

    y_lower, y_upper = 0,0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[cluster_labels]
        c_silhouette_vals.sort()
        y_upper += len(c_silhouette_vals)
        color = cmap(float(i) / n_clusters)
        plt.barh(range(y_lower, y_upper), c_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)
        yticks.append((y_lower + y_upper) / 2)
        y_lower += len(c_silhouette_vals)
        
    silhouette_avg = np.mean(silhouette_vals)

    plt.yticks(yticks, cluster_labels+1)

    # 縦軸はすべての値における
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.title("Silouette for K-means")
    plt.show()


if __name__ == '__main__':
    main()