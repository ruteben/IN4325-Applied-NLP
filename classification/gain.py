from scipy.stats import entropy
import classification
from scipy.stats import entropy
import numpy as np

data = classification.get_data()


def get_column(data, column_index):
    column = []
    length_data = len(data)
    for row in range(0, length_data):
        column.append(data[row][column_index])
    return column


def calc_gains(data):
    gains = []
    for column_index in range(0, data.shape[1]):
        column = np.array(get_column(data, column_index))
        gain = entropy(column)
        gains.append(gain)
    return gains


print(calc_gains(data))