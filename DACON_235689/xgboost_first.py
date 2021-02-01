# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 00:04:28 2021

@author: newave986.git
"""

# 1.18692
# 아무것도 안 건드리고 그냥 배운대로 XGBoost 사용한 코드

import pandas as pd

train=pd.read_csv('train_features.csv')
train_label=pd.read_csv('train_labels.csv')
test=pd.read_csv('test_features.csv')
submission=pd.read_csv('sample_submission.csv')

features = ['id', 'acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']
X_train = train[features].groupby('id').agg(['max', 'min', 'mean'])
X_test = test[features].groupby('id').agg(['max', 'min', 'mean'])

y_train = train_label['label']

from xgboost import XGBClassifier

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(X_train, y_train)

w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]

y_pred = xgb_wrapper.predict_proba(X_test)

submission.iloc[:,1:] = y_pred
submission

submission.to_csv('xgboost.csv', index=False)



