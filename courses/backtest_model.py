import json
# from tools.StochRSI import StochRSI
from util.KDJ import KDJ
import matplotlib.pyplot as plt

"""
 KDJ(9, 3, 3)
开仓：  
        if self.k_buffer[-2] <= self.d_buffer[-2] and self.k_buffer[-2] <= 20 and self.k_buffer[-1] >= self.d_buffer[-1]:
            open_side = 'buy'
        elif self.k_buffer[-2] >= self.d_buffer[-2] and self.k_buffer[-2] >= 80 and self.k_buffer[-1] <= self.d_buffer[-1]:
            open_side = 'sell'

开仓信号完全由KDJ决定
信号：开  检查仓位， 如果有仓位，按仓位加仓， 如果和上次信号相反， 则把所有仓位平掉，在新的方向开仓
5分钟未成交，撤单
当天结束：如果有仓位，则按最后一分钟的开盘价平仓         未考虑能否平掉
超过下午3点， 有信号也不开仓


self.add_profit = 0  # 开仓劣势价， 比如开多，此时开盘价100 ， add_profit = 1 则挂单价： 100 - 1*0.005
self.stop_profit = {0: 0, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7:20}  # 止盈价 1档仓位 20跳止盈， 2档仓位：。。。 
self.stop_loss = 20  # 止损价        超过20跳止损， 未考虑能否平掉     
self.add_position = 4  # 比上次开仓优于N跳， 加仓     
self.one_hand = {0: 1, 1: 2, 2: 4, 3: 4, 4: 4, 5: 4, 6:4}    0档 开1手， 1档开2手， 2档开4手。。。
self.tick_price = 0.005     

"""


class BackTest(object):
    def __init__(self):
        self.data_list = []
        self.tech_index = KDJ(9, 3, 3)
        # self.tech_index = StochRSI(9, 9, 3, 3)
        self.position = {
            'position': 0,  # 开仓数量
            'ave_price': 0,  # 开仓平均价
            'price': [],  # 历次开仓价格
            'volume': 0
        }
        self.today_profit = 0  # 当日盈亏
        self.profit = 0  # 总盈亏
        # 订单记录
        self.orders = {}
        # 参数设置
        self.add_profit = 1  # 开仓劣势价
        self.stop_profit = {0: 0, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20}  # 止盈价
        # self.stop_profit = {0: 0, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10}  # 止盈价
        # self.stop_profit = {0: 0, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4}  # 止盈价
        self.stop_loss = 20  # 止损价
        self.add_position = 4  # 比上次开仓优于N跳， 加仓
        # self.one_hand = {0: 1, 1: , 2: 4, 3: 4, 4: 4, 5: 4, 6: 4}
        self.one_hand = {0: 1, 1: 2, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4}
        self.tick_price = 0.005
        self.init_param()
        self.record = []

    def init_param(self):
        """
        初始化开仓参数
        :return:
        """
        one_day = int(4.5 * 60)
        # 初始化数据
        # with open("record.json", 'r') as load_f:
        # with open("data/5min_data.json", 'r') as load_f:
        with open("record_t9999.json", 'r') as load_f:
            self.data_list = json.load(load_f)
        # 参数初始化长度
        data_length = one_day * 96
        # 初始化指标
        for data in self.data_list[:data_length]:
            self.tech_index.cal_index(data)
        self.data_list = self.data_list[data_length:]
        print('start:', self.data_list[0])

    def position_judge(self, open_side, data):
        if data.get('_id').split('T')[1] > '15-00-00Z':
            print('Over time, not open', data.get('_id'))
            return
        if open_side == 'buy':
            # 平掉之前的仓位，并开新仓
            if self.position.get('position') == 0:
                self.orders = {
                    'price': data.get('O') - self.add_profit * 0.005,
                    'time': data.get('_id'),
                    'side': open_side,
                    'count': 0,
                    'state': 'open'
                }
                print(data.get('_id'), open_side, 'open price', data.get('O'), 'order price', self.orders.get('price'))
            elif self.position.get('position') < 0:
                # 平仓
                profit = (self.position.get('ave_price') - data.get('O')) * abs(self.position.get('position'))
                print(data.get('_id'), 'signal', 'close sell', data.get('O'), '*' * 20, profit)
                self.profit += profit
                self.position = {'position': 0, 'ave_price': 0, 'price': [], 'volume':0}
                self.orders = {
                    'price': data.get('O') - self.add_profit * 0.005,
                    'time': data.get('_id'),
                    'side': open_side,
                    'count': 0,
                    'state': 'open'
                }
                print(data.get('_id'), open_side, 'open price', data.get('O'), 'order price', self.orders.get('price'))
            else:
                # 加仓
                if self.position.get('price')[-1] - data.get('O') >= self.tick_price * self.add_position:
                    self.orders = {
                        'price': data.get('O') - self.add_profit * 0.005,
                        'time': data.get('_id'),
                        'side': open_side,
                        'count': 0,
                        'state': 'open'
                    }
                    print(data.get('_id'), open_side, 'again', data.get('O'), self.orders.get('price'))
                else:
                    print(data.get('_id'), open_side, 'again, but not satisfy price', data.get('O'), self.position.get('price')[-1])
        elif open_side == 'sell':
            if self.position.get('position') == 0:
                self.orders = {
                    'price': data.get('O') + self.add_profit * 0.005,
                    'time': data.get('_id'),
                    'side': open_side,
                    'count': 0,
                    'state': 'open'
                }
                print(data.get('_id'), open_side, 'open price', data.get('O'), 'order price', self.orders.get('price'))
            elif self.position.get('position') > 0:
                # 平掉之前的仓位，并开新仓
                profit = (data.get('O') - self.position.get('ave_price')) * abs(self.position.get('position'))
                print(data.get('_id'), 'signal', 'close buy', data.get('O'), '*' * 20, profit)
                self.profit += profit
                self.position = {'position': 0, 'ave_price': 0, 'price': [], 'volume':0}
                self.orders = {
                    'price': data.get('O') + self.add_profit * 0.005,
                    'time': data.get('_id'),
                    'side': open_side,
                    'count': 0,
                    'state': 'open'
                }
                print(data.get('_id'), open_side, 'open price', data.get('O'), 'order price', self.orders.get('price'))
            else:
                # 加仓
                if data.get('O') - self.position.get('price')[-1] >= self.tick_price * self.add_position:
                    self.orders = {
                        'price': data.get('O') + self.add_profit * 0.005,
                        'time': data.get('_id'),
                        'side': open_side,
                        'count': 0,
                        'state': 'open'
                    }
                    print(data.get('_id'), open_side, 'again', data.get('O'), self.orders.get('price'))
                else:
                    print(data.get('_id'), open_side, 'again, but not satisfy price', data.get('O'), self.position.get('price')[-1])

    def get_signal(self, last_data, data):
        self.tech_index.cal_index(last_data)
        open_side = self.tech_index.get_signal()
        if not open_side:
            return
        self.position_judge(open_side, data)

    def deal_judge(self, data):
        if self.orders.get('side') == 'buy' and self.orders.get('state') == 'open':
            if data.get('L') < self.orders.get('price'):
                volume = self.one_hand.get(abs(self.position['position']))
                self.position['ave_price'] = ((self.position['position'] * self.position['ave_price']) + self.orders.get(
                    'price') * volume) / (self.position['position'] + volume)
                self.position['position'] += 1
                self.position['volume'] += volume
                self.position['price'].append(self.orders.get('price'))
                print(data.get('_id'), 'open buy', self.position)
                self.orders = {}
            else:
                self.orders['count'] += 1
                if self.orders['count'] >= 5:
                    print(data.get('_id'), 'cancel buy', self.orders.get('price'))
                    self.orders = {}
        elif self.orders.get('side') == 'sell' and self.orders.get('state') == 'open':
            if data.get('H') > self.orders.get('price'):
                volume = self.one_hand.get(abs(self.position['position']))
                self.position['ave_price'] = (abs((self.position['position']) * self.position['ave_price']) + self.orders.get(
                    'price') * volume) / (abs(self.position['position']) + volume)
                self.position['position'] -= 1
                self.position['volume'] += volume
                self.position['price'].append(self.orders.get('price'))
                print(data.get('_id'), 'open sell', self.position)
                self.orders = {}
            else:
                self.orders['count'] += 1
                if self.orders['count'] >= 5:
                    print(data.get('_id'), 'cancel sell', self.orders.get('price'))
                    self.orders = {}

    def position_deal_judge(self, data):
        # 判断止盈止损
        stop_profit = self.stop_profit[abs(self.position['position'])]
        if self.position['position'] > 0:
            h_price = self.position.get('ave_price') + stop_profit * self.tick_price
            l_price = self.position.get('ave_price') - self.stop_loss * self.tick_price
            if data.get('H') > h_price:
                # 止盈
                # profit = stop_profit * self.tick_price
                profit = stop_profit * self.tick_price * self.position.get('volume')
                print(data.get('_id'), 'close buy', h_price, self.position.get('ave_price'), '*' * 20, profit)
                self.profit += profit
                self.position = {'position': 0, 'ave_price': 0, 'price': [], 'volume':0}
                self.orders = {}
            elif data.get('L') < l_price:
                # 止损
                # profit = -self.stop_loss * self.tick_price
                profit = -self.stop_loss * self.tick_price* self.position.get('volume')
                print(data.get('_id'), 'close buy', l_price, self.position.get('ave_price'), '*' * 20, profit)
                self.profit += profit
                self.position = {'position': 0, 'ave_price': 0, 'price': [], 'volume':0}
                self.orders = {}
        elif self.position['position'] < 0:
            l_price = self.position.get('ave_price') - stop_profit * self.tick_price
            h_price = self.position.get('ave_price') + self.stop_loss * self.tick_price
            if data.get('L') < l_price:
                # 止盈
                profit = stop_profit * self.tick_price
                print(data.get('_id'), 'close buy', l_price, self.position.get('ave_price'), '*' * 20, profit)
                self.profit += profit
                self.position = {'position': 0, 'ave_price': 0, 'price': [], 'volume':0}
                self.orders = {}
            elif data.get('H') > h_price:
                # 止损
                profit = -self.stop_loss * self.tick_price
                print(data.get('_id'), 'close buy', h_price, self.position.get('ave_price'), '*' * 20, profit)
                self.profit += profit
                self.position = {'position': 0, 'ave_price': 0, 'price': [], 'volume':0}
                self.orders = {}

    def close_today(self, data):
        if self.position['position'] != 0:
            if self.position['position'] < 0:
                profit = -(data.get('O') - self.position['ave_price']) * abs(self.position['position'])
            else:
                profit = (data.get('O') - self.position['ave_price']) * abs(self.position['position'])
            print('close today', data.get('O'), profit)
            self.profit += profit
            self.position = {'position': 0, 'ave_price': 0, 'price': [], 'volume':0}
            self.orders = {}

    def back_test(self):
        for i in range(1, len(self.data_list)):
            self.get_signal(self.data_list[i - 1], self.data_list[i])
            self.deal_judge(self.data_list[i])
            self.position_deal_judge(self.data_list[i])
            if self.data_list[i].get('_id').split('T')[1] == '15-14-00Z':
                self.close_today(self.data_list[i])
                data = self.data_list[i]
                self.record.append({'time': data.get('O'), 'profit': self.profit})
                print('*' * 60, self.data_list[i].get('_id'), self.profit)
        print(self.profit)
        self.draw_pic()

    def draw_pic(self):
        time_list = []
        profit_list = []
        for value in self.record:
            time_list.append(value.get('time'))
            profit_list.append(value.get('profit'))
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('profit')
        plt.subplots_adjust(wspace=0.5)
        # ema_line, = ax.plot(profit_list, color='black')
        plt.plot(profit_list)
        # c_line, = ax.plot(c, color='black')
        plt.show()


if __name__ == '__main__':
    bt = BackTest()
    bt.back_test()
