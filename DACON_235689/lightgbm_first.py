# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 00:22:27 2021

@author: newave986.git
"""


# 아무것도 안 건드리고 그냥 배운대로 LightGBM 사용한 코드

import pandas as pd

train=pd.read_csv('train_features.csv')
train_label=pd.read_csv('train_labels.csv')
test=pd.read_csv('test_features.csv')
submission=pd.read_csv('sample_submission.csv')

features = ['id', 'acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']
X_train = train[features].groupby('id').agg(['max', 'min', 'mean'])
X_test = test[features].groupby('id').agg(['max', 'min', 'mean'])

y_train = train_label['label']

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# ftr = train.data
# target = train.target

# X_train, X_test, y_train, y_test = train_test_split(ftr, target, test_size=0.2, random_state = 10)

lgbm_wrapper = LGBMClassifier(n_estimators=400)
# evals = [(X_test, y_train)]
# logloss로 평가하는 것만 없애면 돌아감.
lgbm_wrapper.fit(X_train.values, y_train)

preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]

y_pred = lgbm_wrapper.predict_proba(X_test)

submission.iloc[:,1:] = y_pred
submission

submission.to_csv('lightgbm.csv', index=False)























