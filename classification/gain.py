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
    amount_of_columns = data.shape[1]
    for column_index in range(0, amount_of_columns):
        column = np.array(get_column(data, column_index))
        gain = info_gain.info_gain(labels, column)
        gains.append([column_index, gain])
    return gains


def output_gains(gains):
    output_file = open('../gains_results.txt', 'w')
    print(gains)
    gains.sort(key=lambda x: x[1], reverse=True)
    print(gains)

    for index in range(0, len(gains)):
        output_file.write("feature %s, gain %.5f \n" % (gains[index][0], gains[index][1]))
    output_file.close()


gains = calc_gains(data)
output_gains(gains)