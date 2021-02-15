# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 02:52:57 2021

@author: newave956.git
"""

import pandas as pd

train=pd.read_csv('train_features.csv')
train_label=pd.read_csv('train_labels.csv')
test=pd.read_csv('test_features.csv')
submission=pd.read_csv('sample_submission.csv')

features = ['id', 'acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']

def q1(x):
    return x.quantile(0.25)

def q2(x):
    return x.quantile(0.75)

grouped = train[features].groupby('id')
X_train = grouped.agg(['max', 'min', 'mean', q1, q2])
X_test = test[features].groupby('id').agg(['max', 'min', 'mean', q1, q2])
y_train = train_label['label']

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.3, max_depth=3, min_child_weight=5, gamma=0.3)
xgb_wrapper.fit(X_train, y_train)

random_grid = {
 'subsample':[i/10.0 for i in range(1,10)],
 'colsample_bytree':[i/10.0 for i in range(1,10)]
}

xgb_random = RandomizedSearchCV(xgb_wrapper, param_distributions = random_grid, cv=5, n_jobs=1)
xgb_random.fit(X_train, y_train)

print('최적 하이퍼 파라미터: \n', xgb_random.best_params_)


