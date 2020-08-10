import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# df = pd.read_excel("data/test.xlsx")
# print(df)
# x = df.index
# fig, ax = plt.subplots(dpi=100, figsize=(9,6))
# ax.set_xticks(range(len(x)))
# ax_xticklabels = map(lambda x:str(x)[0:4]+str(x)[5:7]+str(x)[8:10], x)
# ax.set_xticklabels(ax_xticklabels)
# fig.autofmt_xdate()
#
# plt.bar(df['DATE'], df['sh'])
#
# fig, ax = plt.subplots(dpi=100, figsize=(9,6))
# x = df.index
# ax.plot(range(20), df['cruve'], color = 'b', label=u'累计收益率', alpha = .5)
# ax.set_xticks(range(len(x)))
# ax_xticklabels = map(lambda x:str(x)[0:4]+str(x)[5:7]+str(x)[8:10], x)
# ax.set_xticklabels(ax_xticklabels)
# fig.autofmt_xdate()
# ax.set_xlim(-0.5, len(x)-0.5)

# # 读取数据
# df = pd.read_excel("data/test.xlsx", encoding = 'gbk')
# # 对时间进行一下处理
# num = 5
# time = df['DATE']
#
# plt.xticks(time[::len(time)//num].index, time[::len(time)//num], rotation=30)
# plt.plot(df['DATE'], df['sh'])
# plt.show()

# df = pd.read_excel("data/test.xlsx", encoding = 'gbk')
# fig, ax = plt.subplots(dpi=100, figsize=(9,6))
# x = df['DATE']
# ax.plot(x, df['sh'], color = 'b', label=u'累计收益率', alpha = .5)
# # ax.plot(range(32), df['sh'], color = 'b', label=u'累计收益率', alpha = .5)
# ax.set_xticks(range(len(x)))
# ax_xticklabels = map(lambda x:str(x)[0:4]+str(x)[5:7]+str(x)[8:10], x)
# ax.set_xticklabels(ax_xticklabels)
# fig.autofmt_xdate()
# plt.show()

N=4
df = pd.read_excel("data/test.xlsx", encoding = 'gbk')

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
a =df['DATE']
x_list = range(len(df['DATE']))
y_list = [el.isoformat()[:10] for el in df['DATE']]
ax.set_xticks(x_list)
# ax.xaxis.set_major_locator(x_list.MultipleLocator(6))
ax.set_xticklabels(y_list, rotation=45)
fig.autofmt_xdate()
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=N))

plt.bar(x_list, df['sh'])
plt.show()

#
# N=4
# df = pd.read_excel("data/test.xlsx", encoding = 'gbk')
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(111)
# a =df['DATE']
# length = 4
# # length = len(a) // N
# print(length)
# ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=length))
#
# # c = ax.get_xticklabels()
# # d = ax.get_xticklabels()
# # d = ax.get_xticklabels()[::20]
# x_list = range(len(df['DATE']))
# y_list = [el.isoformat()[:10] for el in df['DATE']]
# ax.set_xticks(range(0, len(x_list), length))
# # ax.xaxis.set_major_locator(x_list.MultipleLocator(6))
# ax.set_xticklabels(y_list[::length], rotation=45)
#
# # for label in ax.get_xticklabels():
# #     label.set_visible(False)
# # for label in ax.get_xticklabels()[::20]:
# #     label.set_visible(True)
#
#
# # ax.set_xticklabels(y_list, rotation=45)
# # ax.set_xticklabels(y_list, rotation=45)
# fig.autofmt_xdate()
# plt.bar(x_list, df['sh'])
# plt.show()

# ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=len(df) // 20))
# for label in ax.get_xticklabels():
#     label.set_visible(False)
# for label in ax.get_xticklabels()[::20]:
#     label.set_visible(True)

# ax.set_xticks(range(0, len(x), len(df) // N))
# ax_xticklabels = list(map(lambda x: str(x)[0:4] + str(x)[5:7] + str(x)[8:10], x))
# ax.set_xticklabels(ax_xticklabels[::len(df) // N])

