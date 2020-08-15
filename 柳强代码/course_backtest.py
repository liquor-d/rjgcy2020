import os
import json
import numpy as np
import pandas as pd
from util.tool_datetime import ToolDateTime
import math
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot, row, column

"""
1. 多空分算
2. 信号平仓
"""


class SingleBackTest(object):
    def __init__(self, param):
        self.test_data = param.get('data')
        self.open_param = param.get('open')
        self.close_param = param.get('close')
        self.side = param.get('side')
        self.factor = param.get('factor')
        self.result = {}
        self.acc_ret = 0
        self.day_record = {}
        self.per_record = []
        self.init_asset = 200000
        self.slippage = 0.4
        self.pos_record = {'pos': 0, 'price': 0, 'side': ''}
        self.is_trade = False
        self.last_ret = 0
        self.time_record = []
        self.has_action = False
        self.multiple_factor = 1    # 合约乘数
        self.hold_days = 10

    def predict_action(self, kline):
        pred_y = kline.get(self.factor)
        if self.side == 'long':
            if pred_y < self.open_param and self.pos_record['pos'] == 0:
                return 'buy'
            elif pred_y > self.close_param and self.pos_record['pos'] == 1:
                return 'close'
        # else:
        #     if pred_y > self.open_param and self.pos_record['pos'] == 0:
        #         return 'sell'
        #     elif pred_y < self.close_param and self.pos_record['pos'] == -1:
        #         return 'close'
        return None

    def update_per_record(self, kline):
        if self.pos_record['side'] == 'buy':
            ret = self.rr(kline.get('C') - self.pos_record['price'], 2)
            hold_time = (kline.get('time').timestamp()  - self.pos_record['time'].timestamp()) / 60
            self.per_record.append([ret, hold_time, ['buy', self.pos_record['price'], self.pos_record['time']], ['sell', kline.get('C'),
                                                                                                                 kline.get('time')]])
            self.acc_ret += ret
            self.pos_record['pos'] = 0
            self.pos_record['side'] = ''
            self.is_trade = True
        elif self.pos_record['side'] == 'sell':
            ret = self.rr(self.pos_record['price'] - kline.get('C'), 2)
            hold_time = (kline.get('time').timestamp()  - self.pos_record['time'].timestamp()) / 60
            self.per_record.append([ret, hold_time, ['sell', self.pos_record['price'], self.pos_record['time']], ['buy', kline.get('C'),
                                                                                                                  kline.get('time')]])
            self.acc_ret += ret
            self.pos_record['pos'] = 0
            self.pos_record['side'] = ''
            self.is_trade = True


    def cal_position(self, kline, action):
        if action == 'buy' and self.pos_record['pos'] == 0:
            self.pos_record['price'] = kline.get('C') + self.slippage
            self.pos_record['side'] = action
            self.pos_record['pos'] = 1
            self.pos_record['time'] = kline.get('time')
            self.is_trade = True
        elif action == 'sell' and self.pos_record['pos'] == 0:
            self.pos_record['price'] = kline.get('C') - self.slippage
            self.pos_record['side'] = action
            self.pos_record['time'] = kline.get('time')
            self.pos_record['pos'] = -1
            self.is_trade = True
        elif action == 'close':
            self.update_per_record(kline)

    def stat(self, kline, flag=False):
        # open close return day annual.ret drawdown mar hold.time num avg.pnl turnover sharpe annual.num
        # 总回报，交易天数，年回报， 回撤， 开平仓次数 平均利润 平均持仓时间 sharp 胜率 盈亏比  最近回撤（最高点到当前最新净值的回撤幅度）
        # 每次交易需要统计的： [利润，持仓时间]
        # 每日需要统计的： sharp [当日结算利润]
        if flag or (self.pos_record['pos'] != 0 and (kline.get('time').timestamp() - self.pos_record['time'].timestamp()) / (3600 * 24) >=
                    self.hold_days):
            self.update_per_record(kline)
            self.is_trade = True
        self.result['total_days'] = self.result.get('total_days', 0) + 1
        if self.is_trade:
            self.result['trade_days'] = self.result.get('trade_days', 0) + 1
        if self.pos_record['side'] == 'buy':
            ret = kline.get('C') - self.pos_record['price']
        elif self.pos_record['side'] == 'sell':
            ret = self.pos_record['price'] - kline.get('C')
        else:
            ret = 0
        self.day_record[kline.get('time')] = self.acc_ret + ret - self.last_ret
        self.last_ret = ret
        self.acc_ret = 0

    def output(self):
        # 输出统计结果
        ret_list = [self.day_record[key] * self.multiple_factor for key in sorted(self.day_record.keys())]  # 每日回报
        if not ret_list:
            return None
        self.time_record = list(self.day_record.keys())
        self.time_record.insert(0, self.test_data[0].get('time'))
        ret_list.insert(0, self.init_asset)
        # 每日净值回报
        for i in range(1, len(ret_list)):
            ret_list[i] = ret_list[i] + ret_list[i - 1]
        net_ret = [ret_list[i] / self.init_asset for i in range(len(ret_list))]  # 每日净值
        max_dd = self.MaxDrawdown(net_ret)
        diff = [(net_ret[i] - net_ret[i - 1]) / abs(net_ret[i - 1]) for i in range(1, len(net_ret))]  # 每日净值较前一日的收益
        if len(self.per_record) == 0:
            return
        self.cal_ave_ret(ret_list)
        self.result['open_param'] = self.open_param
        self.result['close_param'] = self.close_param
        self.result['side'] = self.side
        self.result['net_ret'] = net_ret
        self.result['time_record'] = self.time_record
        self.result['per_record'] = self.per_record     # 用来打买卖点位置
        self.result['total_ret'] = ret_list[-1] - self.init_asset  # 总回报
        self.result['draw_back'] = max_dd  # 最大回撤
        if self.result['total_ret'] > 0:
            self.result['sharp'] = self.rr(np.sqrt(252) * abs((np.average(diff))) / np.std(diff), 3)  # 夏普 每日收益率平均值/每日收益率标准差  每日收益率：
            # 较前一日的收益/前一日的收益
        else:
            self.result['sharp'] = self.rr(-np.sqrt(252) * abs((np.average(diff))) / np.std(diff), 3)
        self.result['calmar'] = self.rr((net_ret[-1] - 1) * 252 / (self.result['total_days'] * max_dd), 3)  # 收益率/最大回撤
        self.result['latest_dr'] = self.rr((1 - net_ret[-1] / max(net_ret)), 3)  # 最近回撤 最高点到当前最新净值的回撤幅度
        return self.result

    def cal_ave_ret(self, ret_list):
        ave_ret = 0  # 平均回报
        ave_hold_time = 0
        action_count = len(self.per_record)  # 开平了多少次
        acc_profit = 0
        acc_p_count = 0
        acc_loss = 0
        acc_l_count = 0
        for record in self.per_record:
            ave_ret += record[0]
            ave_hold_time += record[1]
            if record[0] > 0:
                acc_profit += record[0]
                acc_p_count += 1
            elif record[0] < 0:
                acc_l_count += 1
                acc_loss += record[0]
        if action_count:
            ave_hold_time = self.rr(ave_hold_time / action_count, 2)
            ave_ret = self.rr(ave_ret * self.multiple_factor / action_count, 2)
            self.result['ave_ret'] = self.rr((ret_list[-1] - self.init_asset) / action_count, 2)  # 最后净值/次数
            self.result['ave_ret1'] = ave_ret  # 每次开平/次数
            self.result['win_ratio'] = self.rr(acc_p_count / action_count, 3)  # 胜率
            if acc_p_count == 0 or acc_l_count == 0 or acc_loss == 0:
                self.result['profit_loss'] = 0  # 盈亏比  acc profit/acc loss
            else:
                self.result['profit_loss'] = self.rr(-(acc_profit / acc_p_count) / (acc_loss / acc_l_count), 3)  # 盈亏比  acc profit/acc loss
        else:
            ave_hold_time = ave_hold_time
            self.result['ave_ret'] = ret_list[-1] - self.init_asset
            self.result['ave_ret1'] = ave_ret
            self.result['win_ratio'] = 1  # 胜率
            self.result['profit_loss'] = 0  # 盈亏比  acc profit/acc loss
        self.result['action_count'] = action_count  # 开平动作
        self.result['ave_hold_time'] = ave_hold_time  # 平均持仓时间
        self.result['acc_profit'] = self.rr(acc_profit, 2)  # 所有盈利相加
        self.result['acc_loss'] = self.rr(acc_loss, 2)  # 所有亏损相加

    @staticmethod
    def MaxDrawdown(return_list):
        '''最大回撤率'''
        i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
        if i == 0:
            return -1
        j = np.argmax(return_list[:i])  # 开始位置
        mdd = (return_list[j] - return_list[i]) / (return_list[j])
        if mdd == 0:
            return -1
        else:
            return mdd

    def run(self):
        """

        :param n:  n 分钟K线
        :param m: 预测m 分钟IC
        :return:
        """
        for i, kline in enumerate(self.test_data):
            action = self.predict_action(kline)
            self.cal_position(kline, action)
            if i == len(self.test_data) - 1:
                self.stat(kline, True)
            else:
                self.stat(kline)
            self.is_trade = False
        return self.output()

    # def run_train(self, data_path, factors, out_path):
    #     data = pd.read_csv('data/factor_data.csv', index_col=0,parse_dates=['time'])
    #     list_data = data.to_dict(orient='records')
    #     x = f"x_2"
    #     print(x)
    #     # min_data, max_data = data[x].quantile(0.005), data[x].quantile(0.995)
    #     min_data, max_data = -2.3, 2.3
    #     print('max:', max_data, 'min:', min_data)
    #     open_p = min_data
    #     close_p = max_data
    #     # for side in ['long', 'short']:
    #     # side = 'long'
    #     side = 'short'
    #     bt = SingleBackTest({'data': list_data, 'open': open_p, 'close': close_p, 'side': side, 'factor': x})
    #     result = bt.run()
    #     self.analysis_and_drawpic(result)
    #     print(result)

    @staticmethod
    def rr(x, n=2):
        return round(x, n)

    def draw_pic_tool(self, key, value):
        net_ret = value.get('net_ret')
        net_ret_x = []
        time_record = value.get('time_record')
        # time_record.insert(0, value.get('time_record')[0].replace('T14', 'T9'))
        kline_x = []
        kline_y = []
        buy_action_x = []
        buy_action = []
        sell_action_x = []
        sell_action = []
        for kline in value.get('data'):
            # kline_x.append(ToolDateTime.string_to_datetime(kline.get('time')))
            kline_x.append(kline.get('time'))
            kline_y.append(kline.get('C'))
        for tmp in time_record:
            net_ret_x.append(tmp)
        for record in value.get('per_record'):
            if record[2][0] == 'buy':
                buy_action.append(record[2][1])
                buy_action_x.append((record[2][2]))
            else:
                sell_action.append(record[2][1])
                sell_action_x.append((record[2][2]))
            if record[3][0] == 'buy':
                buy_action.append(record[3][1])
                buy_action_x.append((record[3][2]))
            else:
                sell_action.append(record[3][1])
                sell_action_x.append((record[3][2]))
        p = figure(tools="crosshair, pan, wheel_zoom, xwheel_pan, ywheel_pan, box_zoom, reset, undo, redo, save",
                   title=f"{key}_{value.get('total_ret')}", x_axis_label='time', y_axis_label='quote', width=1200,
                   height=400,
                   x_axis_type='datetime')
        p.xaxis.major_label_orientation = math.pi / 2
        p.line(kline_x, kline_y, line_color='black', legend="kline")
        # p.line(net_ret_x, y, legend="net", line_color="green")
        p.circle(buy_action_x, buy_action, legend="buy", fill_color="red", line_color="red", size=6)
        p.circle(sell_action_x, sell_action, legend="sell", fill_color="blue", line_color="blue", size=6)
        p1 = figure(tools="crosshair, pan, wheel_zoom, xwheel_pan, ywheel_pan, box_zoom, reset, undo, redo, save",
                    title=f"{key}_{value.get('total_ret')}", x_axis_label='time', y_axis_label='net_ret', width=1200,
                    height=400,
                    x_axis_type='datetime')
        p1.line(net_ret_x, net_ret, legend="net", line_color="green")
        return [p, p1]

    def analysis_drawpic(self, result):
        # 选最好的画图 带买卖点
        result['data'] = self.test_data
        tmp = f"{result['side']}_{result['open_param']}_{result['close_param']}"
        fig = self.draw_pic_tool(tmp, result)
        output_file(f"data/pics/result_{tmp}_{ToolDateTime.get_date_string('s')}.html")
        show(column(fig))

if __name__ == '__main__':
    factor = 'y_5_pred'
    # factor = 'y_pred'
    data = pd.read_csv('data/factor_data.csv', index_col=0, parse_dates=['time'])
    list_data = data.to_dict(orient='records')
    side = 'long'
    # side = 'short'

    # min_data, max_data = data[factor].quantile(0.005), data[factor].quantile(0.5)
    # min_data, max_data = data[factor].quantile(0.005), data[factor].quantile(0.995)
    # min_data, max_data = data[factor].quantile(0.02), data[factor].quantile(0.5)
    min_data, max_data = data[factor].quantile(0.02), data[factor].quantile(0.98)
    # min_data, max_data = -2.3, 2.3
    # min_data, max_data = -0.006223011786818557, 0.007941826451176213
    print('max:', max_data, 'min:', min_data)
    open_p = min_data
    close_p = max_data
    bt = SingleBackTest({'data': list_data, 'open': open_p, 'close': close_p, 'side': side, 'factor': factor})

    result = bt.run()
    bt.analysis_drawpic(result)
    result.pop('time_record')
    result.pop('data')
    result.pop('per_record')
    result.pop('net_ret')
    df = pd.DataFrame(result,index=[0])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_rows', None)
    print(df)



