# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 19:23:58 2021

@author: newave986.git
"""

import pandas as pd
import numpy as np

train=pd.read_csv('train_features.csv')
train_label=pd.read_csv('train_labels.csv')
test=pd.read_csv('test_features.csv')
submission=pd.read_csv('sample_submission.csv')

features = ['id', 'acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']

def q1(x):
    return x.quantile(0.25)

def q3(x):
    return x.quantile(0.75)

def IQR(x):
    return q3(x) - q1(x)

def RMS(x):
    return np.sqrt(np.mean(x**2))

def ZCR(x):
        return float("{0:.2f}".format((((np.array(x)[:-1] * np.array(x)[1:]) < 0).sum())/len(x)))

grouped = train[features].groupby('id')
X_train = grouped.agg(['max', 'min', 'mean', 'mad', q1, q3, IQR, RMS, ZCR])
X_test = test[features].groupby('id').agg(['max', 'min', 'mean', 'mad', q1, q3, IQR, RMS, ZCR])
y_train = train_label['label']

from xgboost import XGBClassifier

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.3, max_depth=3, 
                            min_child_weight=5, gamma=0.3,
                            subsample=0.9, colsample_bytree=0.4)
xgb_wrapper.fit(X_train, y_train)

w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]

y_pred = xgb_wrapper.predict_proba(X_test)

submission.iloc[:,1:] = y_pred
submission

submission.to_csv('xgboost_q1q3_iqrrmszcr.csv', index=False)






