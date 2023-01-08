# 事前に pip install imbalanced-learn しておく
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


def main():
    # 単純化のために2つの特徴でデータセットを生成する
    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                            
    n_redundant=0, weights=[0.9, 0.1], random_state=2017)
    print ("Positive class: ", y.tolist().count(1))
    print ("Negative class: ", y.tolist().count(0))

    # ランダムアンダーサンプリング
    rus = RandomUnderSampler()
    X_RUS, y_RUS = rus.fit_resample(X, y)

    # ランダムオーバーサンプリング
    ros = RandomOverSampler()
    X_ROS, y_ROS = ros.fit_resample(X, y)

    # SMOTE
    sm = SMOTE()
    X_SMOTE, y_SMOTE = sm.fit_resample(X, y)

    # 元データとリサンプリング結果を比較
    plt.figure(figsize=(10, 6))
    plt.subplot(2,2,1)
    plt.scatter(X[y==0,0], X[y==0,1], marker='o', color='blue')
    plt.scatter(X[y==1,0], X[y==1,1], marker='+', color='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Original: 1=%s and 0=%s' %(y.tolist().count(1), y.tolist().count(0)))
    plt.subplot(2,2,2)
    plt.scatter(X_RUS[y_RUS==0,0], X_RUS[y_RUS==0,1], marker='o', color='blue')
    plt.scatter(X_RUS[y_RUS==1,0], X_RUS[y_RUS==1,1], marker='+', color='red')
    plt.xlabel('x1')
    plt.ylabel('y2')
    plt.title('Random Under-sampling: 1=%s and 0=%s' %(y_RUS.tolist().count(1), y_RUS.tolist().count(0)))
    plt.subplot(2,2,3)
    plt.scatter(X_ROS[y_ROS==0,0], X_ROS[y_ROS==0,1], marker='o', color='blue')
    plt.scatter(X_ROS[y_ROS==1,0], X_ROS[y_ROS==1,1], marker='+', color='red')
    plt.xlabel('x1')
    plt.ylabel('x2') 
    plt.title('Random over-sampling: 1=%s and 0=%s' %(y_ROS.tolist().count(1), y_ROS.tolist().count(0)))
    plt.subplot(2,2,4)
    plt.scatter(X_SMOTE[y_SMOTE==0,0], X_SMOTE[y_SMOTE==0,1], marker='o', color='blue')
    plt.scatter(X_SMOTE[y_SMOTE==1,0], X_SMOTE[y_SMOTE==1,1], marker='+', color='red')
    plt.xlabel('x1')
    plt.ylabel('y2')
    plt.title('SMOTE: 1=%s and 0=%s' %(y_SMOTE.tolist().count(1), y_SMOTE.tolist().count(0)))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()