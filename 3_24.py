"""
正則化
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics


def main():
    # データの読み込み
    df = pd.read_csv('Data/Grade_Set_2.csv')
    df.columns = ['x','y']

    for i in range(2,50):               # 1乗は処理の必要なし
        colname = 'x_%d'%i              # 列名は「x_べき数」となる
        df[colname] = df['x']**i

    independent_variables = list(df.columns)
    independent_variables.remove('y')

    print(df.head())

    X= df[independent_variables]       # 独立変数
    y= df.y                            # 従属変数

    # 学習データとテストデータの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80, random_state=1)

    # リッジ回帰
    lr = linear_model.Ridge(alpha=0.001)
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    print("------ Ridge Regression ------")
    print("Train MAE: ", metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(metrics.mean_squared_error(y_train, 
    y_train_pred)))
    print("Test MAE: ", metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, 
    y_test_pred)))
    print("Ridge Coef: ", lr.coef_)

    # ラッソ回帰
    lr = linear_model.Lasso(alpha=0.001)
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    print("----- LASSO Regression -----")
    print("Train MAE: ", metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
    print("LASSO Coef: ", lr.coef_)


if __name__ == '__main__':
    main()