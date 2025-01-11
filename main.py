import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

file_path = '/Users/weston/Desktop/Quant/ford_row_stock_data.csv'
row_data = pd.read_csv(file_path)

train_num = int(row_data.shape[0] * 0.8)
val_num = int(row_data.shape[0] * 0.1)
test_num = int(row_data.shape[0] * 0.1)

extracted_raw_data = row_data.iloc[:,2:7]
scaler = StandardScaler()
final_data = scaler.fit_transform(extracted_raw_data)
final_data_df = pd.DataFrame(final_data, columns=extracted_raw_data.columns)

def window_size(dataset, start_point, end_point, step_length, window_size):
    if end_point is None:
        end_point = len(dataset)

    data = []
    labels = []

    while start_point + window_size < end_point:
        data.append(dataset[start_point:start_point + window_size])
        labels.append(dataset[start_point + window_size, 3])  # Extract the fourth value as the label
        start_point += step_length

    return np.array(data), np.array(labels)

x_train, y_train = window_size(dataset=final_data, start_point=0, end_point=train_num, step_length=1, window_size=30)
x_val, y_val = window_size(dataset=final_data, start_point=train_num, end_point=train_num + val_num, step_length=1, window_size=30)
x_test, y_test = window_size(dataset=final_data, start_point=train_num + val_num, end_point=train_num + val_num + test_num, step_length=1, window_size=30)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

sample = next(iter(train_ds))
inputs_shape = sample[0].shape[1:]

inputs = keras.Input(shape=inputs_shape)

x = layers.LSTM(units=8, dropout=0.5, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
x = layers.LeakyReLU()(x)
x = layers.LSTM(units=16, dropout=0.5, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.LeakyReLU()(x)
x = layers.LSTM(units=32, dropout=0.5, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1)(x)  # Single value output

model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001), loss=tf.keras.losses.MeanAbsoluteError())

epochs = 1000

history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
model.evaluate(test_ds)

model.save('/Users/weston/Desktop/Quant/epoachs1000_window30.keras')
np.save('/Users/weston/Desktop/Quant/epoachs1000_window30_history.npy', history.history)



