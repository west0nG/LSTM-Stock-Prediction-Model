import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = keras.models.load_model('./model/E2000W15.keras')

# Load the training history
history = np.load('./model/E2000W15.npy', allow_pickle=True).item()

window = 15

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

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

x_test, y_test = window_size(dataset=final_data, start_point=train_num + val_num, end_point=train_num + val_num + test_num, step_length=1, window_size=window)


predictions = model.predict(x_test)

# Extract the actual "close" values (assuming "close" is the 6th column in the original dataset)
actual_close_values = final_data_df.iloc[train_num + val_num + window:train_num + val_num + test_num + window, 3].values

# Plot the comparison
plt.figure(figsize=(14, 7))
plt.plot(actual_close_values, label='Actual Close Values')
plt.plot(predictions, label='Predicted Close Values')
plt.xlabel('Time Steps')
plt.ylabel('Close Values')
plt.title('Actual vs Predicted Close Values')
plt.legend()
plt.show()

