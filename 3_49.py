from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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


if __name__ == '__main__':
    main()