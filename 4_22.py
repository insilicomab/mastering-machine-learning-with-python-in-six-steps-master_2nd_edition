"""
ハイパーパラメータ調整のためのグリッドサーチ
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def main():
    seed = 2017
    # データの読み込み
    df = pd.read_csv("data/Diabetes.csv")
    X = df.iloc[:,:8].values     # 独立変数
    y = df['class'].values       # 従属変数

    # 正規化
    X = StandardScaler().fit_transform(X)

    # 訓練データセットとテストデータセットに分けてモデルを評価
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    kfold = model_selection.StratifiedKFold(n_splits=5)
    num_trees = 100
    clf_rf = RandomForestClassifier(random_state=seed).fit(X_train, y_train)
    rf_params = {
        'n_estimators': [100, 250, 500, 750, 1000],
        'criterion':  ['gini', 'entropy'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [1, 3, 5, 7, 9]
    }

    # verbose = 10を設定するとタスクが10回完了するごとに進捗状況が表示される
    grid = GridSearchCV(clf_rf, rf_params, scoring='roc_auc', cv=kfold, verbose=10, n_jobs=-1)
    grid.fit(X_train, y_train)
    print ('Best Parameters: ', grid.best_params_)

    results = model_selection.cross_val_score(grid.best_estimator_, X_train,y_train, cv=kfold)
    print ("Accuracy - Train CV: ", results.mean())
    print ("Accuracy - Train : ", metrics.accuracy_score(grid.best_estimator_.predict(X_train), y_train))
    print ("Accuracy - Test : ", metrics.accuracy_score(grid.best_estimator_.predict(X_test), y_test))


if __name__ == '__main__':
    main()