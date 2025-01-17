import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
file_path = './data/ford_row_stock_data.csv'
row_data = pd.read_csv(file_path)

# Split the data into training, validation, and testing sets
train_num = int(row_data.shape[0] * 0.8)
val_num = int(row_data.shape[0] * 0.1)
test_num = int(row_data.shape[0] * 0.1)

# Extract and scale the relevant columns
extracted_raw_data = row_data.iloc[:, 2:7]
scaler = StandardScaler()
final_data = scaler.fit_transform(extracted_raw_data)
final_data_df = pd.DataFrame(final_data, columns=extracted_raw_data.columns)

# Define the window_size function
def window_size(dataset, start_point, end_point, step_length, window_size):
    if end_point is None:
        end_point = len(dataset)

    data = []

    while start_point + window_size <= end_point:
        data.append(dataset[start_point:start_point + window_size])
        start_point += step_length

    return np.array(data)

# Prepare the test data for each model
model1_test_data = window_size(final_data, train_num + val_num, None, 1, 60)
model2_test_data = window_size(final_data, train_num + val_num + 29, None, 1, 30)
model3_test_data = window_size(final_data, train_num + val_num + 44, None, 1, 15)

# Load the models
model1 = tf.keras.models.load_model('./model/E1000W60.keras')
model2 = tf.keras.models.load_model('./model/E2000W30.keras')
model3 = tf.keras.models.load_model('./model/E2000W15.keras')

# Predict using the models
pred1 = model1.predict(model1_test_data).flatten()
pred2 = model2.predict(model2_test_data).flatten()
pred3 = model3.predict(model3_test_data).flatten()

# Ensure the predictions have the same length
min_length = min(len(pred1), len(pred2), len(pred3))
pred1 = pred1[:min_length]
pred2 = pred2[:min_length]
pred3 = pred3[:min_length]

# Calculate the final prediction


temp_array = np.zeros((pred1.shape[0], final_data.shape[1]))
temp_array[:, 3] = pred1  
pred1 = scaler.inverse_transform(temp_array)[:, 3]
#print(pred1)

temp_array = np.zeros((pred2.shape[0], final_data.shape[1]))
temp_array[:, 3] = pred2  
pred2 = scaler.inverse_transform(temp_array)[:, 3]
#print(pred2)

temp_array = np.zeros((pred3.shape[0], final_data.shape[1]))
temp_array[:, 3] = pred3 
pred3 = scaler.inverse_transform(temp_array)[:, 3]


final_pred_real = 0.1140983 * pred1 + 0.60110426 * pred2 + 1.0882431 * pred3 - 5.8132088675939295
#print("Final Pred Data is")
#print(final_pred_real)

test_data_real = row_data.iloc[train_num+val_num+60:, 5]
#print (test_data_real)

test_data_open = row_data.iloc[train_num+val_num+60:, 2]


min_length = min(len(final_pred_real), len(test_data_real), len(test_data_open))
final_pred_real = final_pred_real[:min_length]
test_data_real = test_data_real[:min_length]
test_data_open = test_data_open[:min_length]

compare_data = np.column_stack((final_pred_real, test_data_real,test_data_open))

print(compare_data)

#Strategy

Capital = 10000
Capital_Progress = []
Profit_Count = []
Accumulated_Capital_Count = []
Counter = 0

#print(len(compare_data))

#and compare_data[i,0] > compare_data[i,2]
for i in range(1,len(compare_data) ):
    if compare_data[i,0] > compare_data[i-1,1]:
        Counter = Capital/compare_data[i,2]*compare_data[i,1]
        Profit = Counter - Capital
        Capital = Counter
        Capital_Progress.append(Capital)
        Profit_Count.append(Profit)
    else:
        Capital_Progress.append(Capital)
        Profit_Count.append(0)


# Actual Close Data on the test set
Actual_Close = row_data.iloc[train_num + val_num + 60:, 5].values * 30

plt.figure(figsize=(10, 6))
plt.plot(Profit_Count, label='Profit')
plt.xlabel('Days')
plt.ylabel('Profit')
plt.plot(Actual_Close, label='Actual Close')
plt.title('Profit')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(Capital_Progress, label='Capital Progress')
plt.xlabel('Days')
plt.ylabel('Capital')
plt.plot()
plt.title('Capital Progress')
plt.legend()
plt.show()


