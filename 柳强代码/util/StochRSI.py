class StochRSI(object):
    def __init__(self, n, m, p1, p2):
        """
        LC := REF(CLOSE,1); //REF(C,1) 上一周期的收盘价
        RSI:= SMA(MAX(CLOSE-LC,0),N,1)/SMA(ABS(CLOSE-LC),N,1) *100;
        STOCHRSI:= MA(RSI-LLV(RSI,M),P1)/MA(HHV(RSI,M)-LLV(RSI,M),P1)*100;  LLV（l,60）表示：检索60天内的最低价
        ％D: MA(STOCHRSI,P2);  天数，作平滑
        :param n, m, p1, p2:
        n:RSI的alpha，
        m: stochRSI, 最低和最高的的长度
        p1: stochRSI 的 ma 长度
        p2: stochRSI
        """
        self.n = n
        self.alpha = 1 / n
        self.buffer_length = m
        self.p1 = p1
        self.p2 = p2
        self.last_data = None
        self.last_ema = None
        self.up = 0
        self.down = 0
        self.last_ema_up = 0
        self.last_ema_down = 0
        self.ema_up = None
        self.ema_down = None
        self.rs = None
        self.rsi = None
        self.stochrsi = None
        self.buffer_rsi = []
        self.tmp_l = []
        self.h_l = []
        self.stochrsi_buffer = []
        self.D = None
        self.accuracy = 8

    def copy_srsi(self):
        srsi = StochRSI(self.n, self.buffer_length, self.p1, self.p2)
        srsi.last_data =self.last_data
        srsi.last_ema = self.last_ema
        srsi.up = self.up
        srsi.down = self.down
        srsi.last_ema_up = self.last_ema_up
        srsi.last_ema_down = self.last_ema_down
        srsi.ema_down = self.ema_down
        srsi.ema_up = self.ema_up
        srsi.rs = self.rs
        srsi.rsi = self.rsi
        srsi.stochrsi = self.stochrsi
        srsi.D = self.D
        srsi.buffer_rsi = self.copy_list(self.buffer_rsi)
        srsi.tmp_l = self.copy_list(self.tmp_l)
        srsi.h_l = self.copy_list(self.h_l)
        srsi.stochrsi_buffer = self.copy_list(self.stochrsi_buffer)
        return srsi

    def copy_list(self, data_list):
        tmp = []
        for i in data_list:
            tmp.append(i)
        return tmp

    def smma(self):
        """
        计算smma
        :return:
        """
        # 这里可以调整，从第1日算 还是第N日算 smma
        if not self.last_ema_up:
            self.ema_up = round(self.up, self.accuracy)
            self.ema_down = round(self.down, self.accuracy)
            self.last_ema_up = self.ema_up
            self.last_ema_down = self.ema_down
        else:
            self.ema_up = round(self.up * self.alpha + self.last_ema_up * (1 - self.alpha), self.accuracy)
            self.ema_down = round(self.down * self.alpha + self.last_ema_down * (1 - self.alpha), self.accuracy)
            self.last_ema_up = self.ema_up
            self.last_ema_down = self.ema_down

    def cal_up_down(self, data):
        """
        计算较前一日的涨跌
        :param data:
        :return:
        """
        if data - self.last_data > 0:
            self.up = data - self.last_data
            self.down = 0
        else:
            self.up = 0
            self.down = self.last_data - data

    def ma(self, data):
        j = 0
        for i in data:
            j += i
        return j / len(data)

    def cal_index(self, data):
        # 初始化第一个数字
        close_price = float(data.get('C'))
        if not self.last_data:
            self.last_data = close_price
            return None
        self.cal_up_down(close_price)
        self.smma()
        if self.ema_down != 0:
            self.rs = round(self.ema_up / self.ema_down, self.accuracy)
            self.rsi = round(100 * self.rs / (1 + self.rs), self.accuracy)
            if len(self.buffer_rsi) < self.buffer_length:
                self.buffer_rsi.append(self.rsi)
            else:
                self.buffer_rsi.append(self.rsi)
                self.buffer_rsi.pop(0)
                # MA(RSI - LLV(RSI, M), P1) / MA(HHV(RSI, M) - LLV(RSI, M), P1) * 100;
                self.tmp_l.append(self.rsi - min(self.buffer_rsi))
                self.h_l.append(max(self.buffer_rsi) - min(self.buffer_rsi))
                if len(self.tmp_l) > self.p1:
                    self.tmp_l.pop(0)
                if len(self.h_l) > self.p1:
                    self.h_l.pop(0)
                if self.ma(self.h_l) == 0:
                    self.stochrsi = 100
                else:
                    self.stochrsi = round(self.ma(self.tmp_l) / self.ma(self.h_l) * 100, self.accuracy)
                self.stochrsi_buffer.append(self.stochrsi)
                if len(self.stochrsi_buffer) > self.p2:
                    self.stochrsi_buffer.pop(0)
                self.D = round(self.ma(self.stochrsi_buffer), self.accuracy)
        self.last_data = close_price

    def get_signal(self):
        if self.stochrsi_buffer[-2] < 10 and self.stochrsi_buffer[-1] > 10:
            open_side = 'buy'
            return open_side
            # 平掉之前的仓位，并开新仓
        elif self.stochrsi_buffer[-2] > 90 and self.stochrsi_buffer[-1] < 90:
            open_side = 'sell'
            return open_side
        return None


if __name__ == '__main__':
    data_list = [5.76, 6.34, 6.97, 7.67, 8.44, 9.28, 8.52, 8.11]
    r = StochRSI(6,14,3,3)
    for i in data_list:
        r.cal_index(i)
