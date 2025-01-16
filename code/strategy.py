import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

train_num = int(row_data.shape[0] * 0.8)
val_num = int(row_data.shape[0] * 0.1)
test_num = int(row_data.shape[0] * 0.1)

file_path = './data/ford_row_stock_data.csv'
row_data = pd.read_csv(file_path)