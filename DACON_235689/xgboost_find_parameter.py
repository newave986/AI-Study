# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 01:33:16 2021

@author: newave986.git
"""

# xgboost with groupby quantile and hyper parameter 
# 결과: 'min_child_weight': 1, 'max_depth': 6

import pandas as pd

os.getcwd()

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

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=5)
xgb_wrapper.fit(X_train, y_train)

w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]

random_grid = {
 'max_depth':range(3,10,3),
 'min_child_weight':range(1,6,2)
}

xgb_random = RandomizedSearchCV(xgb_wrapper, param_distributions = random_grid, cv=2, n_jobs=1)
xgb_random.fit(X_train, y_train)
print('최적 하이퍼 파라미터: \n', xgb_random.best_params_)







