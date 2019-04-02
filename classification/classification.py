"""This is the main file for performing classification, containing all steps necessary"""

import xgboost as xgb
from sklearn.datasets import load_boston
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import jsonlines
import csv


# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
#
#
# print(dtrain)




with open('../../NLP_data/preprocessed.csv') as file:
    datafile = csv.reader(file, delimiter=";")
    data = []
    i = 0
    for row in datafile:
        i = i + 1
        if row and i > 1:  # I have the feeling it skips 2i every time here but I don't know why
            row_no_id = row[1:]
            row_int = []
            for value in row_no_id:
                row_int.append(float(value))
            data.append(row_int)

data = np.array(data)

truth = []
with jsonlines.open('../../NLP_data/truth.jsonl') as json_file:
    for obj in json_file:
        if obj['truthClass'] == 'clickbait':
            truth.append(1)
        else:
            truth.append(0)

truth = np.array(truth)

size_train = round(0.8*len(data))

data_train = data[:size_train]
data_test = data[size_train:]

labels_train = truth[:size_train]
labels_test = truth[size_train:]

dtrain = xgb.DMatrix(data_train, label=labels_train)
dtest = xgb.DMatrix(data_test, label=labels_test)

param = {
    'max_depth': 5,  # the maximum depth of each tree,
    'min_child_weight': 3,
    'gamma': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1,
    'eta': 1,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    # 'objective': 'binary:logistic'
    'objective': 'multi:softprob',
    'num_class': 2
}

bst = xgb.train(param, dtrain)
# make prediction
preds = bst.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])

# print(best_preds)
# print(precision_score(y_test, best_preds, average='macro'))
# print(recall_score(y_test, best_preds, average='macro'))
# print(accuracy_score(y_test, best_preds))


print(best_preds)
print(np.size(best_preds))
print(np.count_nonzero(best_preds))
print(precision_score(labels_test, best_preds, average='macro'))
print(recall_score(labels_test, best_preds, average='macro'))
print(accuracy_score(labels_test, best_preds))
# print(roc_auc_score(labels_test, best_preds))


def get_data():
    with open('../../NLP_data/preprocessed.csv') as file:
        datafile = csv.reader(file, delimiter=";")
        data = []
        i = 0
        for row in datafile:
            i = i + 1
            if row and i > 1:  # I have the feeling it skips 2i every time here but I don't know why
                row_no_id = row[1:]
                row_int = []
                for value in row_no_id:
                    row_int.append(float(value))
                data.append(row_int)

    return np.array(data)


def get_labels():
    truth = []
    with jsonlines.open('../../NLP_data/truth.jsonl') as json_file:
        for obj in json_file:
            if obj['truthClass'] == 'clickbait':
                truth.append(1)
            else:
                truth.append(0)

    return np.array(truth)

def create_DMatrices(data, labels, test_size):


def train_model(dtrain, dtest, params):


def


def parameter_sweep():
