"""
ハイパーパラメータ調整のためのグリッドサーチ
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from bayes_opt.util import Colours
from sklearn.ensemble import RandomForestClassifier as RFC


def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    """
    ランダムフォレストのクロスバリデーション

    この関数はn_estimators、min_samples_split、max_featuresをパラメータとして，ランダムフォレスト分類器をインスタンス化する。これにデータとターゲットを組み合わせてクロスバリデーションを行う。ここでの我々の目標はlog lossを最小化するn_estimators, min_samples_split, max_featuresの組み合わせを見つけることである。
    """
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2
    )

    cval = cross_val_score(
        estimator,
        data, 
        targets,
        scoring='neg_log_loss', 
        cv=4
    )

    return cval.mean()


class Optimization():
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    

    def _rfc_crossval(self, n_estimators, min_samples_split, max_features):
        """ 
        RandomForestクロスバリデーションのラッパー
        n_estimatorsとmin_samples_splitを渡す前に，integerにキャストしていることに注目してほしい。さらにmax_featuresが(0, 1)の範囲外の値を取ることを避けるためにそれに応じてキャップされていることも確認している
        """
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=self.data,
            targets=self.targets,
        )
    

    def optimize_rfc(self):
        optimizer = BayesianOptimization(
            f=self._rfc_crossval,
            pbounds={
                "n_estimators": (10, 250),
                "min_samples_split": (2, 25),
                "max_features": (0.1, 0.999),
            },
            random_state=1234,
            verbose=2
        )
        optimizer.maximize(n_iter=10)
        print("Final result:", optimizer.max)
        
        return optimizer


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
    
    print(Colours.green("--- Optimizing Random Forest ---"))
    optimization = Optimization(X_train, y_train)
    optimization.optimize_rfc()


if __name__ == '__main__':
    main()