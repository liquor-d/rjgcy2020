import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import *


def fun4_1():
    """
    r_5 x_2   -0.06728628
    :return:
    """
    df = pd.read_csv('data/factor_data.csv', index_col=0)
    x = df['x_2'].values.reshape(-1,1)
    y = df['r_5'].values.reshape(-1,1)
    model = LinearRegression()
    # 拟合模型
    model.fit(x, y)
    print(model)
    # coef = -0.0023351  0.00022241
    print(model.coef_)
    print(model.intercept_)
    # 预测
    y_pred = model.predict(np.reshape(x, (-1, 1)))
    df['y_5_pred'] = y_pred
    df.to_csv('data/factor_data.csv')
    train_score = model.score(x, y)
    print("r2", train_score)
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y)
    plt.plot(x, y_pred, color="r")
    plt.show()

def fun4_2():
    df = pd.read_csv('data/factor_data.csv', index_col=0)
    print( df['y_10_pred'].min(), df['y_10_pred'].max())
    print( df['y_10_pred'].quantile(0.005), df['y_10_pred'].quantile(0.995))
    print( df['r_10'].quantile(0.005), df['r_10'].quantile(0.995))


def fun4_3():
    df = pd.read_csv('data/factor_data.csv', index_col=0)
    x = df['x_2'].values.reshape(-1, 1)
    y = df['r_5'].values.reshape(-1, 1)
    # 拟合模型
    # coef = -0.0023351  0.00022241
    coef = -0.025351
    intercept = 0.0032241
    y_pred = coef*x + intercept
    df['y_pred'] = y_pred
    df.to_csv('data/factor_data.csv')
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y)
    plt.plot(x, y_pred, color="r")
    plt.show()


fun4_1()
