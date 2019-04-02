"""This is the main file for performing classification, containing all steps necessary"""

import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import jsonlines
import csv

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
    size_train = round((1-test_size) * len(data))

    data_train = data[:size_train]
    data_test = data[size_train:]

    labels_train = labels[:size_train]
    labels_test = labels[size_train:]

    dtrain = xgb.DMatrix(data_train, label=labels_train)
    dtest = xgb.DMatrix(data_test, label=labels_test)

    return [dtrain, dtest, labels_test]


def train_model(dtrain, dtest, labels_test, params):
    bst = xgb.train(params, dtrain)
    preds = bst.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    accuracy = accuracy_score(labels_test, best_preds)
    recall = recall_score(labels_test, best_preds, average='macro')
    precision = precision_score(labels_test, best_preds, average='macro')

    # print("Number of posts classified as clickbait: %s" % np.count_nonzero(best_preds))
    # print("precision: %s" % precision)
    # print("recall: %s" % recall)
    # print("accuracy: %s" % accuracy)

    return [precision, recall, accuracy]


def parameter_sweep(dtrain, dtest, labels_test):
    accuracy = 0;
    recall = 0;
    precision = 0;

    accuracy_params = [0, 0, 0, 0, 0]
    recall_params = [0, 0, 0, 0, 0]
    precision_params = [0, 0, 0, 0, 0]

    max_depth_poss = np.arange(1, 10, 1)

    for max_depth in max_depth_poss:
        print(max_depth)
        params = {
            'max_depth': max_depth,  # the maximum depth of each tree,
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

        [precision_new, recall_new, accuracy_new] = train_model(dtrain, dtest, labels_test, params)

        if precision_new > precision:
            precision = precision_new
            precision_params = [max_depth, 0, 0, 0, 0]

        if accuracy_new > accuracy:
            accuracy = accuracy_new
            accuracy_params = [max_depth, 0, 0, 0, 0]

        if recall_new > recall:
            recall = recall_new
            recall_params = [max_depth, 0, 0, 0, 0]

    return[accuracy, precision, recall, accuracy_params, recall_params, precision_params]


# params = {
#     'max_depth': 5,  # the maximum depth of each tree,
#     'min_child_weight': 3,
#     'gamma': 2,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'scale_pos_weight': 1,
#     'eta': 1,  # the training step for each iteration
#     'silent': 1,  # logging mode - quiet
#     # 'objective': 'binary:logistic'
#     'objective': 'multi:softprob',
#     'num_class': 2
# }
#
# [accuracy, precision, recall] = train_model(params)

data = get_data()
labels = get_labels()

[dtrain, dtest, labels_test] = create_DMatrices(data, labels, 0.2)

[accuracy, precision, recall, accuracy_params, recall_params, precision_params] = parameter_sweep(dtrain, dtest, labels_test)

print("best accuracy: %s" % accuracy)
print("with params: %s" % accuracy_params)
print("")
print("best precision: %s" % precision)
print("with params: %s" %  precision_params)
print("")
print("best recall: %s" % recall)
print("with params: %s" % recall_params)

