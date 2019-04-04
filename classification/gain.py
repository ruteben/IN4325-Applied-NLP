from scipy.stats import entropy
import classification

data = classification.get_data()

arr = [1, 2, 5, 2, 5, 0, 0, 10, 5, 7, 2]
entr = entropy(arr)
print(entr)


