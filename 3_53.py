
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


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

    # 凝集性クラスター
    model = AgglomerativeClustering(n_clusters=3)
    
    # リスト3-49でインポートした虹彩データセットにモデルを当てはめてみよう
    model.fit(X)
    print(model.labels_)
    iris['pred_species'] =  model.labels_

    print("Accuracy :", metrics.accuracy_score(iris.species, iris.pred_species))
    print("Classification report :", metrics.classification_report(iris.species, iris.pred_species))


if __name__ == '__main__':
    main()