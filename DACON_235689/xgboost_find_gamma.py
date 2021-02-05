# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:16:25 2021

@author: newave986.git
"""

# gamma 튜닝

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

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6) # max_depth=6임을 찾았으므로 이제는 계속 6으로 둔다.
xgb_wrapper.fit(X_train, y_train)

w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]

random_grid = {
 'gamma':[i/10.0 for i in range(0,5)]
}

xgb_random = RandomizedSearchCV(xgb_wrapper, param_distributions = random_grid, cv=5, n_jobs=1)
xgb_random.fit(X_train, y_train)
print('최적 하이퍼 파라미터: \n', xgb_random.best_params_)

