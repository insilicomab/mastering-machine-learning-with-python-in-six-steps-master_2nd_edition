# 事前に pip install imbalanced-learn しておく
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split


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

    X_RUS_train, X_RUS_test, y_RUS_train, y_RUS_test = train_test_split(X_RUS, y_RUS, test_size=0.3, random_state=2017)
    X_ROS_train, X_ROS_test, y_ROS_train, y_ROS_test = train_test_split(X_ROS, y_ROS, test_size=0.3, random_state=2017)
    X_SMOTE_train, X_SMOTE_test, y_SMOTE_train, y_SMOTE_test = train_test_split(X_SMOTE, y_SMOTE, test_size=0.3, random_state=2017)

    # 決定木の構築
    clf = tree.DecisionTreeClassifier(random_state=2017)
    clf_rus = clf.fit(X_RUS_train, y_RUS_train)
    clf_ros = clf.fit(X_ROS_train, y_ROS_train)
    clf_smote = clf.fit(X_SMOTE_train, y_SMOTE_train)

    # モデルの性能を評価
    print ("\nRUS - Train AUC : ",metrics.roc_auc_score(y_RUS_train, clf.predict(X_RUS_train)))
    print ("RUS - Test AUC : ",metrics.roc_auc_score(y_RUS_test, clf.predict(X_RUS_test)))
    print ("ROS - Train AUC : ",metrics.roc_auc_score(y_ROS_train, clf.predict(X_ROS_train)))
    print ("ROS - Test AUC : ",metrics.roc_auc_score(y_ROS_test, clf.predict(X_ROS_test)))
    print ("\nSMOTE - Train AUC : ",metrics.roc_auc_score(y_SMOTE_train, clf.predict(X_SMOTE_train)))
    print ("SMOTE - Test AUC : ",metrics.roc_auc_score(y_SMOTE_test, clf.predict(X_SMOTE_test)))


if __name__ == '__main__':
    main()