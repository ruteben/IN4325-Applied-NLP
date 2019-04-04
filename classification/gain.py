from scipy.stats import entropy
import classification

data = classification.get_data()


def get_column(data, column_index):
    column = []
    length_data = len(data)
    for row in range(0, length_data):
        column.append(data[row][column_index])
    return column

a = get_column(data, 1)
print(a)
print(sum(a))

b = get_column(data, 2)
print(b)
print(sum(b))

c = get_column(data, 3)
print(c)
print(sum(c))


def calc_gains(data):
    gains = []
    for column_index in range(0, data.shape[1]):
        column = get_column(data, column_index)
        gain = entropy(column)
        gains.append(gain)
    return gains


print(calc_gains(data))