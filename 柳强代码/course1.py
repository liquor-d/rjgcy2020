import pandas as pd
import sys
import time

"""
1. 清洗整理数据
2. 合成使用的K线

"""
# 1. 去除异常值, 合成干净的数据保存成csv
def fun1_1():
    df = pd.read_csv('data/rawdata.csv')
    print(df.head())
    df = df.dropna()
    print(df.head())
    # # 时间索引会有问题
    # # df.to_csv('data/1min_data.csv')
    names = list(df.columns)
    names[0] = 'time'
    df.columns = names
    print(df.head())
    # # df.to_csv('data/1min_data.csv')
    # # 索引号会有问题
    df = df.reset_index(drop=True)
    print(df.head())
    df.to_csv('data/1min_data.csv')

# fun1_1()


# 2. 合成N分钟K线
def fun1_2(n):
    def merge_kline(kline_list):
        t = kline_list[0]['time']
        time_struct = time.strptime(t, "%Y/%m/%d %H:%M")
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
        data = {'time': start_time, 'open': kline_list[0]['open'], 'close': kline_list[-1]['close']}
        volume = 0
        amount = 0
        for kline in kline_list:
            volume += kline['volume']
            amount += kline['amount']
            data['high'] = data.get('high', 0) if data.get('high', 0) > kline['high'] else kline['high']
            data['low'] = data.get('low', sys.maxsize) if data.get('low', sys.maxsize) < kline['low'] else kline['low']
        data['volume'] = volume
        data['amount'] = amount
        return data

    df = pd.read_csv("data/1min_data.csv", index_col=0)
    data = df.to_dict(orient="records")
    # 看各个字段的长度是否一致，再次检查
    # 合成N分钟K线
    kline_list = []
    results = []
    for i, value in enumerate(data):
        kline_list.append(value)
        if len(kline_list) % n == 0:
            new_kline = merge_kline(kline_list)
            kline_list.clear()
            results.append(new_kline)
    pd.DataFrame(results).to_csv(f'data/{n}min_data.csv')
    print(data)


# 2. 合成N分钟K线
def fun1_3():
    def merge_kline(kline_list):
        t = kline_list[0]['time']
        time_struct = time.strptime(t, "%Y/%m/%d %H:%M")
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
        data = {'time': start_time, 'open': kline_list[0]['open'], 'close': kline_list[-1]['close']}
        volume = 0
        amount = 0
        for kline in kline_list:
            volume += kline['volume']
            amount += kline['amount']
            data['high'] = data.get('high', 0) if data.get('high', 0) > kline['high'] else kline['high']
            data['low'] = data.get('low', sys.maxsize) if data.get('low', sys.maxsize) < kline['low'] else kline['low']
        data['volume'] = volume
        data['amount'] = amount
        return data

    df = pd.read_csv("data/1min_data.csv", index_col=0)
    data = df.to_dict(orient="records")
    # 看各个字段的长度是否一致，再次检查
    # 合成N分钟K线
    kline_list = []
    results = []
    first_time = None
    for i, value in enumerate(data):
        if not first_time:
            first_time = value['time']
        if value['time'][:9] == first_time[:9]:
            kline_list.append(value)
        else:
            first_time = value['time']
            new_kline = merge_kline(kline_list)
            kline_list= [value]
            results.append(new_kline)
    pd.DataFrame(results).to_csv(f'data/day_data.csv')
    print(data)




# fun1_1()
# fun1_2(5)
fun1_3()


