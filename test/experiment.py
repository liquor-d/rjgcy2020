# %% test tf

import tensorflow as tf

print(tf.__version__)


# %%
import akshare as ak
test_history_data=ak.stock_zh_a_daily(symbol="sh600582", adjust="hfq")



# %%
test_history_data["close"].plot()




# %%
import plotly.graph_objects as go
from datetime import datetime

fig = go.Figure(data=[go.Candlestick(x=test_history_data.index,
                open=test_history_data["open"],
                high=test_history_data["high"],
                low=test_history_data["low"],
                close=test_history_data["close"])])

fig.show()

# %%


# %%
