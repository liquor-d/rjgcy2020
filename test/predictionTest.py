#%%
from math import sin
import backtrader as bt
import matplotlib.pyplot as plt
import akshare as ak
import tensorflow as tf
import numpy as np
import math

from tensorflow.keras.layers import Input, Dense, Flatten, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pylab as plt
#%% [markdown]

# # Define useful functions

#%% prepare the training data

def getTrain(stock):
    stock.dropna()
    x=stock.iloc[:-1,:].values
    y=stock["close"].values[1:]

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
        x_sliced.append(x[t:t+win_size])
        y_sliced.append(y[t+win_size-1])
    x_sliced=np.array(x_sliced)
    y_sliced=np.array(y_sliced)
    return x_sliced.reshape((-1,win_size,feature_size)),y_sliced

# %% build the model
def build_model(win_size,feature_size):
    i=Input(shape=(win_size,feature_size))
    x=LSTM(50)(i)
    x=Dense(6,activation='tanh')(x)
    x=Dense(1)(x)
    model=Model(i,x)
    model.compile(loss='mse',optimizer=Adam(lr=0.01))
    return model


# %% [markdown]

# # put everything together

#%% load the data

stock_hfq_df = ak.stock_zh_a_daily(symbol="sh600000", adjust="hfq")  # 利用 AkShare 获取后复权数据

x_train,y_train,x_validation,y_validation,x_test,y_test=getTrain(stock_hfq_df)
feature_size=7
plt.plot(x_train[:100,3])
plt.plot(y_train[:100])


# %% sanity check
sin_wav=np.sin(0.1*np.arange(300))+np.random.randn(300)*0.05

x_train=sin_wav[0:-150]
y_train=sin_wav[1:-149]
x_validation=sin_wav[-150:-80]
y_validation=sin_wav[-149:-79]
x_test=sin_wav[-80:-1]
y_test=sin_wav[-79:]

plt.plot(x_train)
plt.plot(y_train)
feature_size=1

# %% slice
win_size=20
x_train,y_train=slice_time_series(x_train,y_train,win_size)
x_validation,y_validation=slice_time_series(x_validation,y_validation,win_size)
x_test,y_test=slice_time_series(x_test,y_test,win_size)
#%%
plt.plot(x_train[:100,win_size-1,0])
plt.plot(y_train[:100])


# %% build so we can train incrementally
model=build_model(win_size,feature_size)
# %% train
result=model.fit(x_train,y_train,
    batch_size=100,
    epochs=100,
    validation_data=(x_validation,y_validation))

# %% plot loss

plt.plot(result.history['loss'],label="loss")
plt.plot(result.history['val_loss'],label="val_loss")
plt.legend()

# %% one step forcast

y_hat=model.predict(x_test)
plt.plot(y_test,label='true')
plt.plot(y_hat,label='prediction')
plt.legend()

# %% multi step
y_hat=[]
x_test_last=x_test[0]
for i in range(x_test.shape[0]):
    y_hat_s1=model.predict(x_test_last.reshape(1,-1,feature_size))[0,0]
    x_test_last=np.roll(x_test_last,-1)
    x_test_last[-1]=y_hat_s1
    y_hat.append(y_hat_s1)
plt.plot(y_test,label='true')
plt.plot(y_hat,label='prediction')
plt.legend()        

# %% calc mase
mase=np.sum(np.abs(y_hat[1:]-y_test[1:]))/np.sum(np.abs(y_test[:-1]-y_test[1:]))
# %%
