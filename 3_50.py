from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

    # K Meansクラスター
    model = KMeans(n_clusters=3, random_state=11)
    model.fit(X)
    print (model.labels_)

    # 教師なしなのでラベルが割り当てられている
    # 実際のラベルと一致しないので，すべての1を0に，0を1に変換する
    # 結果として2が最適に見える
    iris['pred_species'] =  np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
    print ("Accuracy :", metrics.accuracy_score(iris.species, iris.pred_species))
    print ("Classification report :", metrics.classification_report(iris.species, iris.pred_species))

    # プロットのサイズを設定
    plt.figure(figsize=(10,7))

    # 赤，緑，青のカラーマップの作成
    cmap = ListedColormap(['r', 'g', 'b'])

    # セパルをプロット
    plt.subplot(2, 2, 1)
    plt.scatter(iris['sepallength(cm)'], iris['sepalwidth(cm)'], c=cmap(iris.species), marker='o', s=50)
    plt.xlabel('sepallength(cm)')
    plt.ylabel('sepalwidth(cm)')
    plt.title('Sepal (Actual)')

    plt.subplot(2, 2, 2)
    plt.scatter(iris['sepallength(cm)'], iris['sepalwidth(cm)'], c=cmap(iris.pred_species), marker='o', s=50)
    plt.xlabel('sepallength(cm)')
    plt.ylabel('sepalwidth(cm)')
    plt.title('Sepal (Predicted)')

    plt.subplot(2, 2, 3)
    plt.scatter(iris['petallength(cm)'], iris['petalwidth(cm)'], c=cmap(iris.species),marker='o', s=50)
    plt.xlabel('petallength(cm)')
    plt.ylabel('petalwidth(cm)')
    plt.title('Petal (Actual)')

    plt.subplot(2, 2, 4)
    plt.scatter(iris['petallength(cm)'], iris['petalwidth(cm)'], c=cmap(iris.pred_species),marker='o', s=50)
    plt.xlabel('petallength(cm)')
    plt.ylabel('petalwidth(cm)')
    plt.title('Petal (Predicted)')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()