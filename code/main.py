import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint


epochs = 2000
window = 30

file_path = './data/ford_row_stock_data.csv'
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

x_train, y_train = window_size(dataset=final_data, start_point=0, end_point=train_num, step_length=1, window_size=window)
x_val, y_val = window_size(dataset=final_data, start_point=train_num, end_point=train_num + val_num, step_length=1, window_size=window)
x_test, y_test = window_size(dataset=final_data, start_point=train_num + val_num, end_point=train_num + val_num + test_num, step_length=1, window_size=window)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

sample = next(iter(train_ds))
inputs_shape = sample[0].shape[1:]

os.makedirs('./model', exist_ok=True)

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

checkpoint_callback = ModelCheckpoint(
    filepath='./model/E{epoch:04d}W30.keras',
    save_freq=200 * len(train_ds),  # Save every 200 epochs
    save_weights_only=False,
    verbose=1
)

history = model.fit(train_ds, epochs=epochs, validation_data=val_ds,callbacks=[checkpoint_callback])
model.evaluate(test_ds)

model.save('./model/E2000W30.keras')
np.save('./model/E2000W30.npy', history.history)




