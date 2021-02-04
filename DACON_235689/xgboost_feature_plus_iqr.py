# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 02:34:00 2021

@author: newave986.git
"""

import pandas as pd
import os

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

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=5)
xgb_wrapper.fit(X_train, y_train)

w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]

y_pred = xgb_wrapper.predict_proba(X_test)

submission.iloc[:,1:] = y_pred
submission

submission.to_csv('xgboost_q1q2.csv', index=False)






