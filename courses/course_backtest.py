from util.KDJ import KDJ
import pandas as pd


class BackTest():
    def __init__(self):
        self.data = pd.read_csv('data/5min_data.csv')
        self.fee = 0.0001
        self.signal = KDJ(9,3,3)

    def get
    def run(self):
        pass


if __name__ == '__main__':
    bt = BackTest()
    bt.run()