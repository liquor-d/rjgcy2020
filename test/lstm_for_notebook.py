# %%导入必要库
import pandas as pd
import numpy as np  # numpy也是做数据处理比较常用的库，官方文档：https://numpy.org/
import tensorflow as tf  # 我们玩神经网络的库，官方网站：https://tensorflow.google.cn/
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

# %%读数据
df = pd.read_excel('ProcessedData.xlsx')
df = df[29:].set_index(df['time'][29:]).drop(columns=['time', 'Unnamed: 0'])  # 设置时间为索引，把含空值的与不必要的列去掉

tf.random.set_seed(0)  # 初始w和b是随机的，如果不设随机种子会使每次重运行的结果有细微差别，设置随机种子以获取更加稳定的结果

# %%设置训练集、验证集和测试集的比例为7：2：1
n = len(df)
train_split = int(0.7 * n)
val_split = int(0.9 * n)

df_mean = df[:train_split].mean(axis=0)  # 计算每个特征在训练集的均值备用，axis=0表示计算的是每个特征而不是每日各个特征的均值
df_std = df[:train_split].std(axis=0)  # 计算每个特征在训练集的标准差备用
df = (df - df_mean) / df_std  # 标准化

df = df.values

#%%
def window_generator(dataset, target, start_index, end_index,
                     history_size, target_size):
    """
    dataset:collection of features; target:collection of labels;

    start_index:beginning of the slice; end_index:end of the slice;

    history_size:input width; target_size:label width;

    """

    features = []
    labels = []

    start_index += history_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        features.append(dataset[indices])
        labels.append(target[i:i + target_size])

    return np.array(features), np.array(labels)

#%%
def loss_curve(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

#%%
EPOCHS = 100  # 设置最大训练轮数为100轮
EVALUATION_INTERNAL = 120

# 数据增强参数备用
BATCH_SIZE = 100
BUFFER_SIZE = 2000

#%%
def compile_and_fit(model, train_data, val_data, patience=10):
    # 为防止过拟合，监视验证集上的loss值，在10个epoch内没有发生太大变化则终止训练
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=patience,
        mode='auto',
        restore_best_weights=True)  # 返回最优参数，而非训练停止时的参数

    # 模型编译
    model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0),  # 设置优化器
                  loss='mae')  # 设置损失函数

    # 模型拟合
    history = model.fit(train_data, epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERNAL,
                        validation_steps=50,
                        validation_data=val_data,
                        callbacks=[early_stopping])
    return history

#%%
X_train_lstm, y_train_lstm = window_generator(dataset=df, target=df[:, 1], start_index=0,
                                              end_index=train_split, history_size=5, target_size=1)

X_val_lstm, y_val_lstm = window_generator(dataset=df, target=df[:, 1], start_index=train_split,
                                          end_index=val_split, history_size=5, target_size=1)
X_test_lstm, y_test_lstm = window_generator(dataset=df, target=df[:, 1], start_index=val_split,
                                            end_index=n, history_size=5, target_size=1)
train_lstm = tf.data.Dataset.from_tensor_slices((X_train_lstm, y_train_lstm))
train_lstm = train_lstm.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()  # 数据增强，shuffle只对训练集做
val_lstm = tf.data.Dataset.from_tensor_slices((X_val_lstm, y_val_lstm))
val_lstm = val_lstm.cache().batch(BATCH_SIZE).repeat()  # 数据增强，shuffle只对训练集做
test_lstm = tf.data.Dataset.from_tensor_slices((X_test_lstm, y_test_lstm))

lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(1)
])

#%%
lstm_history = compile_and_fit(lstm, train_lstm, val_lstm)
#%%
lstm_results = lstm.predict(X_test_lstm)

fig = plt.figure(figsize=(15, 8))
ax = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(10))

plt.plot(y_test_lstm, label='oringin')
plt.plot(lstm_results, label='lstm')
plt.legend()

plt.show()

#%%
def cal_mase(label, prediction):
    prediction_error_sum = 0
    pre_label_error_sum = 0
    for i in range(len(prediction)):
        prediction_error_sum += abs(label[i] - prediction[i])
    numerator = prediction_error_sum / len(prediction)

    for j in range(len(label) - 1):
        pre_label_error_sum += abs(prediction[j+1] - prediction[j])
    denominator = pre_label_error_sum / (len(label) - 1)
    mase = numerator / denominator
    return mase

#%%
mase_lstm = cal_mase(y_test_lstm, lstm_results)
