import pandas as pd
from util.StochRSI import StochRSI
from util.KDJ import KDJ
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np


def fun3_1():
    """
    计算未来收益
    :return:
    """
    df = pd.read_csv('data/day_data.csv', index_col=0)
    df['r_1'] = (df['close']-df.shift(1)['close'])/df.shift(1)['close']
    df['r_1'] = df['r_1'].shift(-1)

    df['r_5'] = (df['close'] - df.shift(5)['close']) / df.shift(5)['close']
    df['r_5'] = df['r_5'].shift(-5)

    df['r_10'] = (df['close'] - df.shift(10)['close']) / df.shift(10)['close']
    df['r_10'] = df['r_10'].shift(-10)

    df.dropna()
    df.to_csv('data/raw_factor_data.csv')


# fun3_1()


def fun3_2():
    """
    打图看一下收益
    :return:
    """
    df = pd.read_csv('data/raw_factor_data.csv', index_col=0)
    y = (df['close'] - df.shift(10)['close'])
    plt.plot(y)
    plt.show()
    # 发现前期数据异常 可能不能用  去掉
    df = df[100:]
    df = df.reset_index(drop=True)
    df.to_csv('data/factor_data.csv')


# fun3_2()

def fun3_3():
    # 计算因子值
    df = pd.read_csv('data/factor_data.csv', index_col=0)
    df.rename(columns={"high": "H", "low": "L", 'open': 'O', 'close': 'C'}, inplace=True)
    data = df.to_dict(orient="records")
    kdj = KDJ(12,6,3)
    rsi = StochRSI(9,3,3,3)
    for i, kline in enumerate(data):
        kdj.cal_index(kline)
        rsi.cal_index(kline)
        df.loc[i, 'k'] = kdj.K
        df.loc[i, 'rsi'] = rsi.rsi
    df = df[100:-10]
    df = df.reset_index(drop=True)
    df.to_csv('data/factor_data.csv')


# fun3_3()

def fun3_4():
    # 看因子是否平稳  不平稳就要做处理
    df = pd.read_csv('data/factor_data.csv', index_col=0)
    plt.plot(df['k'])
    plt.show()

# fun3_4()

def fun3_5():
    def zscore(x):
        return (x - np.mean(x)) / np.std(x)
    df = pd.read_csv('data/factor_data.csv', index_col=0)
    ss = StandardScaler()
    # 这里用到了未来信息，实际情况是要在训练集或验证集上去做标准化
    df['x_1'] = ss.fit_transform(df['k'].values.reshape(-1, 1))
    df['x_2'] = ss.fit_transform(df['rsi'].values.reshape(-1, 1))
    df.to_csv('data/factor_data.csv')

# fun3_5()

def fun3_6():
    df = pd.read_csv('data/factor_data.csv', index_col=0)
    print("x1 x2", np.corrcoef(df['x_1'], df['x_2'])[0])
    for column in ['r_1', 'r_5', "r_10"]:
        for factor in ['x_1', 'x_2']:
            print(column, factor, np.corrcoef(df[column], df[factor])[0])

# fun3_3()
fun3_6()