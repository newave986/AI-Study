# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 01:51:58 2021

@author: newave986.git
"""

# 기본 코드 - registered, casual 없애고 시도

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

bike_df = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
submission = pd.read_csv('./sampleSubmission.csv')

# set bike_df data 
bike_df['datetime'] = bike_df.datetime.apply(pd.to_datetime)
bike_df['year'] = bike_df.datetime.apply(lambda x: x.year)
bike_df['month'] = bike_df.datetime.apply(lambda x: x.month)
bike_df['day'] = bike_df.datetime.apply(lambda x: x.day)
bike_df['hour'] = bike_df.datetime.apply(lambda x: x.hour)
bike_df.drop(['datetime'], axis=1, inplace=True)
bike_df.drop(['casual', 'registered'], axis=1, inplace=True)

# set test data
test['datetime'] = test.datetime.apply(pd.to_datetime)
test['year'] = test.datetime.apply(lambda x: x.year)
test['month'] = test.datetime.apply(lambda x: x.month)
test['day'] = test.datetime.apply(lambda x: x.day)
test['hour'] = test.datetime.apply(lambda x: x.hour)
test.drop(['datetime'], axis=1, inplace=True)

# get log
bike_df['count'] = np.log1p(bike_df['count'])

# encoding
bike_feature = pd.DataFrame.copy(bike_df)
X_features = bike_feature.drop(['count'], axis=1, inplace=False)

X_features_ohe = pd.get_dummies(X_features, columns=['year','month','hour', 'holiday',
                                              'workingday','season','weather'])

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Lasso

# set features
features = ['year', 'month', 'day', 'hour', 'holiday','workingday',
            'temp', 'atemp', 'humidity', 'windspeed', 'season', 'weather']

# use xgboost
X_train = bike_df[features]
X_test = test[features]
y_train = bike_df['count']

# RANDOMFORESTREGRESSOR
RFR = RandomForestRegressor(n_estimators=500)
RFR.fit(X_train, y_train)
preds = RFR.predict(X_test)

submission.iloc[:,1:] = preds
submission['count'] = np.expm1(submission['count'])
submission

submission.to_csv('./basic_randomforest.csv', index=False)

# GRADIENTBOOSTINGREGRESSOR
GBR = GradientBoostingRegressor(n_estimators=500)
GBR.fit(X_train, y_train)
preds = GBR.predict(X_test)

submission = pd.read_csv('./sampleSubmission.csv')
submission.iloc[:,1:] = preds
submission['count'] = np.expm1(submission['count'])
submission

submission.to_csv('./basic_gradientboostingregressor.csv', index=False)

# XGBOOST
XGB = XGBRegressor(n_estimators=500)
XGB.fit(X_train, y_train)
preds = XGB.predict(X_test)

submission = pd.read_csv('./sampleSubmission.csv')
submission.iloc[:,1:] = preds
submission['count'] = np.expm1(submission['count'])
submission

submission.to_csv('./basic_xgboost.csv', index=False)

# LIGHTGBM
LGBM = LGBMRegressor(n_estimators=500)
LGBM.fit(X_train, y_train)
preds = LGBM.predict(X_test)

submission = pd.read_csv('./sampleSubmission.csv')
submission.iloc[:,1:] = preds
submission['count'] = np.expm1(submission['count'])
submission

submission.to_csv('./basic_lightgbm.csv', index=False)

# LINEARREGRESSION
LR = LinearRegression()
LR.fit(X_train, y_train)
preds = LR.predict(X_test)

submission.iloc[:,1:] = preds
submission['count'] = np.expm1(submission['count'])
submission

submission.to_csv('./basic_linearregression.csv', index=False)

# LASSO
LS = Lasso()
LS.fit(X_train, y_train)
preds = LS.predict(X_test)

submission.iloc[:,1:] = preds
submission['count'] = np.expm1(submission['count'])
submission

submission.to_csv('./basic_lasso.csv', index=False)


