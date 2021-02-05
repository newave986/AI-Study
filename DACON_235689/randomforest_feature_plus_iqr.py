# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:16:25 2021

@author: newave986.git
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

y_train = train_label['label']

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_jobs=-1, random_state=0, min_samples_leaf=30)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)

submission.iloc[:,1:] = y_pred
submission

submission.to_csv('randomforest_q1q2.csv', index=False)

















