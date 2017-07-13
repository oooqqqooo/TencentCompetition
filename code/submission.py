# -*- coding: utf-8 -*-
"""
frx
2017.6.1
"""

import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# load data
data_root = "."
dfTrain = pd.read_csv("%s/train.csv"%data_root)
dfTest = pd.read_csv("%s/test.csv"%data_root)
dfAd = pd.read_csv("%s/userChangeAd.csv"%data_root)
dfUser = pd.read_csv("%s/userChangeAge.csv"%data_root)
dfPosition = pd.read_csv("%s/position.csv"%data_root)

# process data
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTrain = pd.merge(dfTrain, dfUser, on="userID")
dfTrain = pd.merge(dfTrain, dfPosition, on="positionID")
dfTest = pd.merge(dfTest, dfAd, on="creativeID")
dfTest = pd.merge(dfTest, dfUser, on="userID")
dfTest = pd.merge(dfTest, dfPosition, on="positionID")
y_train = dfTrain["label"].values

# feature engineering/encoding
def encoding():
    enc = OneHotEncoder(handle_unknown='ignore')
    feats = ['age','gender','education','appID', 'telecomsOperator', 'marriageStatus','residence','creativeID','haveBaby','hometown','camgaignID','countNum','actionNum',"advertiserID",'positionID']
    for i,feat in enumerate(feats):
        
        if (feat =='countNume' or feat == 'actionNum' or feat == 'age'):
            x_train = dfTrain[feat].values.reshape(-1, 1)
            x_test = dfTest[feat].values.reshape(-1, 1)
        else:
            x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
            x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
        if i == 0:
            X_train, X_test = x_train, x_test
        else:
            X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
    return X_train, X_test

X_train, X_test = encoding()

# model training

#逻辑回归
def LR(X_train, y_train):
    lr = LogisticRegression(solver='newton-cg', max_iter=200)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    proba_test1 = lr.predict_proba(X_test)[:,1]
    return proba_test1

proba_test1 = LR(X_train, y_train)

#xgboost
def XGB(X_train, y_train):
    model = xgb.XGBClassifier(
     learning_rate =0.1,
     n_estimators=2500,
     max_depth=7,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     scoring='logloss',
     nthread=8,
     min_child_weight=2,
     seed=27)
    model.fit(X_train, y_train)
    proba_test2 = model.predict_proba(X_test)[:,1]
    return proba_test2

proba_test2 = XGB(X_train, y_train)

#blending融合
def blending(proba_test1,proba_test2):
    proba_test = []
    for i in range(len(proba_test2)):
        proba_test.append((float(proba_test1[i])+float(proba_test2[i]))/2)
    return proba_test

proba_test = blending(proba_test1,proba_test2)

# submission
def submission(proba_test):
    df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)
    with zipfile.ZipFile("submission.zip", "w") as fout:
        fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
    return None

submission(proba_test)
