### 问题： 给一个时间长度的K线数据，通过预测未来涨跌幅，进行低买高卖，获得收益
+ 通过前人总结的经验，进行模糊判断
+ 通过预测下一时刻的价格，进行交易
+ 通过寻找对未来收益有解释性的因子，辅助交易

#### 通过前人总结的经验，进行模糊判断
+ 比如MACD，KDJ这类指标，即是前人发明的参考工具。
MACD(12,26,9)公式：
DIF:EMA(CLOSE,SHORT)-EMA(CLOSE,LONG);  慢线-快线
DEA:EMA(DIF,MID);                     

KDJ(9,3,3)：
RSV:=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
K:SMA(RSV,M1,1);
D:SMA(K,M2,1);
J:3*K-2*D;


