import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

model1 = tf.keras.models.load_model('./model/E1000W60.keras')
model2 = tf.keras.models.load_model('./model/E2000W30.keras')
model3 = tf.keras.models.load_model('./model/E2000W15.keras')

row_data = pd.read_csv('./data/ford_row_stock_data.csv')

train_num = int(row_data.shape[0] * 0.8)

extracted_raw_data = row_data.iloc[:,2:7]
scaler = StandardScaler()
final_data = scaler.fit_transform(extracted_raw_data)



def window_size(dataset, start_point, end_point, step_length, window_size):
    if end_point is None:
        end_point = len(dataset)

    data = []

    while start_point + window_size < end_point:
        data.append(dataset[start_point:start_point + window_size])
        start_point += step_length

    return np.array(data)

model1_test_data = window_size(final_data,0,train_num,1,60)
model2_test_data = window_size(final_data,29,train_num,1,30)
model3_test_data = window_size(final_data,44,train_num,1,15)

print(model1_test_data.shape)
print(model2_test_data.shape)
print(model3_test_data.shape)

label_data = final_data[60:train_num, 3]

pred1 = model1.predict(model1_test_data).flatten()
pred2 = model2.predict(model2_test_data).flatten()
pred3 = model3.predict(model3_test_data).flatten()

min_length = min(len(pred1), len(pred2), len(pred3))
pred1 = pred1[:min_length]
pred2 = pred2[:min_length]
pred3 = pred3[:min_length]
label_data = label_data[:min_length]

X = np.column_stack((pred1, pred2, pred3))
y= label_data

reg = LinearRegression().fit(X, y)

y_pred = reg.predict(X)

joblib.dump(reg, './model/reg.pkl')








