from sklearn import datasets
import classification

iris = datasets.load_iris()
data_all = iris.data
labels_all = iris.target

data = data_all[:100]
labels = labels_all[:100]

print(data)
print(labels)

classification.run_train_model(data, labels)