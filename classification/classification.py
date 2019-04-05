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
    auc = roc_auc_score(labels_test, best_preds, average="macro")

    # print("Number of posts classified as clickbait: %s" % np.count_nonzero(best_preds))
    # print("precision: %s" % precision)
    # print("recall: %s" % recall)
    # print("accuracy: %s" % accuracy)
    # print("auc: %s" % auc)

    return [precision, recall, accuracy, auc]

# # train model parameter sweep
# def parameter_sweep(dtrain, dtest, labels_test):

# cross validation parameter sweep
def parameter_sweep(data, labels):
    accuracy = 0;
    recall = 0;
    precision = 0;
    auc = 0;

    accuracy_params = [0, 0, 0, 0]
    recall_params = [0, 0, 0, 0]
    precision_params = [0, 0, 0, 0]
    auc_params = [0, 0, 0, 0]

    max_depth_poss = np.arange(3, 10, 1)
    min_child_weight_poss = np.arange(1, 6, 1)
    gamma_poss = np.arange(0.1, 1, 0.1)
    eta_poss = np.arange(0.1, 0.3, 0.1)

    for max_depth in max_depth_poss:
        print("max_depth: %s" % max_depth)
        for min_child_weight in min_child_weight_poss:
            for gamma in gamma_poss:
                for eta in eta_poss:

                    params = {
                        'max_depth': max_depth,  # the maximum depth of each tree,
                        'min_child_weight': min_child_weight,
                        'gamma': gamma,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'scale_pos_weight': 1,
                        'eta': eta,  # the training step for each iteration
                        'silent': 1,  # logging mode - quiet
                        # 'objective': 'binary:logistic'
                        'objective': 'multi:softprob',
                        'num_class': 2
                    }
                    # # train model parameter sweep
                    # [precision_new, recall_new, accuracy_new, auc_new] = train_model(dtrain, dtest, labels_test, params)

                    # cross validation parameter sweep
                    [precision_new, recall_new, accuracy_new, auc_new] = cross_validation(data, labels, params, 5)

                    if precision_new > precision:
                        precision = precision_new
                        precision_params = [max_depth, min_child_weight, gamma, eta]

                    if accuracy_new > accuracy:
                        accuracy = accuracy_new
                        accuracy_params = [max_depth, min_child_weight, gamma, eta]

                    if recall_new > recall:
                        recall = recall_new
                        recall_params = [max_depth, min_child_weight, gamma, eta]

                    if auc_new > auc:
                        auc = auc_new
                        auc_params = [max_depth, min_child_weight, gamma, eta]

    return[accuracy, precision, recall, auc, accuracy_params, recall_params, precision_params, auc_params]


def cross_validation(data, labels, params, folds):
    size_test_set = round(len(data)/folds)
    accuracy = 0
    precision = 0
    recall = 0
    auc = 0

    for fold in range(0, folds):
        size_first_train_part = fold*size_test_set

        first_train_part = data[:size_first_train_part]
        test_set = data[size_first_train_part: size_first_train_part + size_test_set]
        last_train_part = data[size_first_train_part + size_test_set:]
        train_set = np.append(first_train_part, last_train_part, axis=0)

        first_labels_part = labels[:size_first_train_part]
        labels_test = labels[size_first_train_part: size_first_train_part + size_test_set]
        last_labels_part = labels[size_first_train_part + size_test_set:]
        labels_train = np.append(first_labels_part, last_labels_part, axis=0)

        dtrain = xgb.DMatrix(train_set, label=labels_train)
        dtest = xgb.DMatrix(test_set, label=labels_test)

        [precision_new, recall_new, accuracy_new, auc_new] = train_model(dtrain, dtest, labels_test, params)

        accuracy += accuracy_new
        precision += precision_new
        recall += recall_new
        auc += auc_new

    accuracy = accuracy/folds
    precision = precision/folds
    recall = recall/folds
    auc = auc/folds

    return [precision, recall, accuracy, auc]


def run_train_model():
    params = {
        'max_depth': 5,  # the maximum depth of each tree,
        'min_child_weight': 2,
        'gamma': 0.9,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1,
        'eta': 0.4,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        # 'objective': 'binary:logistic'
        'objective': 'multi:softprob',
        'num_class': 2
    }

    data = get_data()
    print(data.shape[0])
    labels = get_labels()

    [dtrain, dtest, labels_test] = create_DMatrices(data, labels, 0.2)

    [accuracy, precision, recall, auc] = train_model(dtrain, dtest, labels_test, params)

    print("precision: %s" % precision)
    print("recall: %s" % recall)
    print("accuracy: %s" % accuracy)
    print("auc: %s" % auc)


def run_parameter_sweep():
    data = get_data()
    labels = get_labels()

    [dtrain, dtest, labels_test] = create_DMatrices(data, labels, 0.2)

    # cross validation parameter sweep
    [accuracy, precision, recall, auc, accuracy_params, recall_params, precision_params, auc_params] = parameter_sweep(
        data, labels)

    # # train model parameter sweep
    # [accuracy, precision, recall, auc, accuracy_params, recall_params, precision_params, auc_params] = parameter_sweep(
    #     dtrain, dtest, labels_test)

    print("")
    print("best accuracy: %s" % accuracy)
    print("with params: %s" % accuracy_params)
    print("")
    print("best precision: %s" % precision)
    print("with params: %s" % precision_params)
    print("")
    print("best recall: %s" % recall)
    print("with params: %s" % recall_params)
    print("")
    print("best auc: %s" % auc)
    print("with params: %s" % auc_params)

    output_file = open('../parameter_sweep_results.txt', 'w')

    output_file.write("params: [max_depth, min_child_weight, gamma, eta] \n \n")
    output_file.write("best accuracy: %s \n" % accuracy)
    output_file.write("with params: %s \n \n" % accuracy_params)
    output_file.write("best precision: %s \n" % precision)
    output_file.write("with params: %s \n \n" % precision_params)
    output_file.write("best recall: %s \n" % recall)
    output_file.write("with params: %s \n \n" % recall_params)
    output_file.write("best auc: %s \n" % auc)
    output_file.write("with params: %s \n \n" % auc_params)

    output_file.close()


def run_cross_validation():
    params = {
        'max_depth': 5,  # the maximum depth of each tree,
        'min_child_weight': 2,
        'gamma': 0.9,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1,
        'eta': 0.4,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        # 'objective': 'binary:logistic'
        'objective': 'multi:softprob',
        'num_class': 2
    }

    data = get_data()
    labels = get_labels()

    [precision, recall, accuracy, auc] = cross_validation(data, labels, params, 5)

    print("precision: %s" % precision)
    print("recall: %s" % recall)
    print("accuracy: %s" % accuracy)
    print("auc: %s" % auc)


# run_parameter_sweep()
