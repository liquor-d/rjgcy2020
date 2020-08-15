#%%
import backtrader as bt
import matplotlib.pyplot as plt
import akshare as ak
import tensorflow as tf
import numpy as np
import math

from tensorflow.keras.layers import Input, Dense, Flatten, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#%% [markdown]

# # Define useful functions

#%% prepare the training data

def getTrain(stock):
    stock.dropna()
    x=stock_hfq_df.iloc[:-1,:].values
    y=stock_hfq_df["close"].values[1:]

    train_ratio=0.5
    validation_ratio=0.2

    train_ends=math.floor(x.shape[0]*train_ratio)
    validation_ends=train_ends+1+math.floor(x.shape[0]*validation_ratio)

    x_train=x[0:train_ends,:]
    y_train=y[0:train_ends].reshape(-1)
    x_validation=x[train_ends:validation_ends,:]
    y_validation=y[train_ends:validation_ends].reshape(-1)
    x_test=x[validation_ends:,:]
    y_test=y[validation_ends:].reshape(-1)
    return x_train,y_train,x_validation,y_validation,x_test,y_test


# %% slice the time series for training
def slice_time_series(x,y,win_size):
# win_size=15
# x=x_train
# y=y_train
    x_sliced=[]
    y_sliced=[]
    for t in range(len(x)-win_size):
        x_slice=x[t:t+win_size]
        x_sliced.append(x_slice)
        y_sliced.append(y[t+win_size-1])
    x_sliced=np.array(x_sliced)
    y_sliced=np.array(y_sliced)
    return x_sliced,y_sliced

# %% build the model
def build_model(win_size):
    i=Input(shape=(win_size,7))
    x=LSTM(30)(i)
    x=Dense(1)(x)
    model=Model(i,x)
    model.compile(loss='mse',optimizer=Adam(lr=0.05))
    return model


# %% [markdown]

# # put everything together

#%% load the data

stock_hfq_df = ak.stock_zh_a_daily(symbol="sh600000", adjust="hfq")  # 利用 AkShare 获取后复权数据

x_train,y_train,x_validation,y_validation,x_test,y_test=getTrain(stock_hfq_df)
# %% slice
win_size=3
x_train,y_train=slice_time_series(x_train,y_train,win_size)
x_validation,y_validation=slice_time_series(x_validation,y_validation,win_size)
x_test,y_test=slice_time_series(x_test,y_test,win_size)


# %% build so we can train incrementally
model=build_model(win_size)
# %% train
result=model.fit(x_train,y_train,
    batch_size=100,
    epochs=10,
    validation_data=(x_validation,y_validation))

# %% plot loss
import matplotlib.pylab as plt
plt.plot(result.history['loss'],label="loss")
plt.plot(result.history['val_loss'],label="val_loss")
plt.legend()

# %% one step forcast

y_hat=model.predict(x_test)
plt.plot(y_test,label='true')
plt.plot(y_hat,label='prediction')
plt.legend()

# %% calc mase
mase=np.sum(np.abs(y_hat[1:]-y_test[1:]))/np.sum(np.abs(y_test[:-1]-y_test[1:]))