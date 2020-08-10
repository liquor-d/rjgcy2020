import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("data/test.xlsx")
print(df)
plt.bar(df['DATE'], df['sh'])
plt.title(u'上证综合指数日成交额')
plt.xlabel('日期', size=1)
plt.ylabel(u'成交金额:上证综合指数')

def format_date(x, pos=None):
    #保证下标不越界,很重要,越界会导致最终plot坐标轴label无显示
    thisind = np.clip(int(x+0.5), 0, N-1)
    return r.date[thisind].strftime('%Y-%m-%d')

fig = plt.figure()
ax.plot(ind, r.adj_close, 'o-')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))

plt.xticks(time[::len(time)//num].index, time[::len(time)//num], rotation=30)
plt.show()
# print(df.head())

# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif']=['SimHei']
# df = pd.read_excel('成交额统计.xlsx','Sheet1')
# data_dict = {}
# for i,j in zip('日期', '成交金额:上证综合指数'):
#     data_dict[i] = j
# fig = plt.figure()
# x_list = [i for i in data_dict.keys()]
# y_list = [i for i in data_dict.values()]
# plt.bar(x_list, y_list)
# plt.title(u'上证综合指数日成交额')
# plt.xlabel('日期', size=1)
# plt.ylabel(u'成交金额:上证综合指数')
# plt.show()



pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install --upgrade tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
