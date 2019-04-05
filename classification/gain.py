from scipy.stats import entropy
import classification
from info_gain import info_gain
import numpy as np

data = classification.get_data()
labels = classification.get_labels()

def get_column(data, column_index):
    column = []
    length_data = len(data)
    for row in range(0, length_data):
        column.append(data[row][column_index])
    return column


def calc_gains(data):
    gains = []
    for column_index in range(0, 1):
        print(column_index)
        column = np.array(get_column(data, column_index))
        gain = info_gain.info_gain(labels, column)
        gains.append(gain)
    return gains


print(calc_gains(data))
