"""This is the main file for performing classification, containing all steps necessary"""

import xgboost as xgb
from sklearn.datasets import load_boston
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_train)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# dtrain = xgb.DMatrix('../../NLP_data/useritemIds_train_itemsplit_fold1of5.csv?format=csv&label_column=0')
# dtest = xgb.DMatrix('../../NLP_data/useritemIds_test_itemsplit_fold3of5.csv?format=csv&label_column=0')

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset

bst = xgb.train(param, dtrain)
# make prediction
preds = bst.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])
print(best_preds)
print(precision_score(y_test, best_preds, average='macro'))
print(recall_score(y_test, best_preds, average='macro'))
print(accuracy_score(y_test, best_preds))
