# -*- coding: utf-8 -*-
'''
frx
2017.6.1
腾讯社交广告算法比赛，LR+XGBoost模型，加各种特征，数据经过自己的处理
userChangeAge.csv，userChangeAd的某些值根据分析重新赋值了，但是效果一般。
'''
import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import scipy as sp

from sklearn.metrics import log_loss

import numpy as np
import scipy.sparse
import xgboost as xgb

import sklearn.preprocessing as preprocessing
# from sklearn.naive_bayes import GaussianNB


# load data
data_root = "."
dfTrain = pd.read_csv("%s/train2.csv"%data_root)
dfTest = pd.read_csv("%s/test2.csv"%data_root)
dfAd = pd.read_csv("%s/userChangeAd.csv"%data_root)
#把hometown和residence重新赋值
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
    enc = OneHotEncoder()
    #scaler = preprocessing.StandardScaler()
    feats = [ 'age','gender','education','appID', 'telecomsOperator', 'marriageStatus','residence','creativeID','haveBaby','hometown','camgaignID','countNum','actionNum',"advertiserID",'positionID']
    for i,feat in enumerate(feats):
        # if (feat == 'countNum' or feat == 'actionNum'):
        #     x_train = scaler.fit_transform(dfTrain[feat])
        #     x_test = scaler.transform(dfTest[feat])
        # else:
        x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
        x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
        if i == 0:
            X_train, X_test = x_train, x_test
        else:
            X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

    return X_train, X_test

X_train,X_test = encoding()

# model training
#在数据集比较大的时候用牛顿法训练参数更好
#朴素贝叶斯
# proba_test = [[]]
# model = BernoulliNB()
# nbStart = time.time()
# model.fit(X_train, y_train)
# nbCostTime = time.time() - nbStart
# proba_test = np.array(model.predict_proba(X_test)[:1])
# print ("朴素贝叶斯建模耗时 %f 秒" %(nbCostTime))


#逻辑回归
def LR(X_train, y_train):
    lr = LogisticRegression(solver='newton-cg', max_iter=200)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    proba_test1 = lr.predict_proba(X_test)[:,1]
    return proba_test1

proba_test1 = LR(X_train, y_train)
#
# print(proba_test1[:3])

# m_regress = xgb.XGBRegressor(n_estimators=1000,max_depth=5,seed=27)

#XGBoost
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

#tt = model.fit(X_train, y_train)

#print('重要性：',model.get_xgb_imp(tt,feats))
#proba_test = model.predict_proba(X_test)[:,1]

# print(proba_test2[:3])
def blending(proba_test1,proba_test2):
    proba_test = []
    for i in range(len(proba_test2)):
        proba_test.append((float(proba_test1[i])*float(proba_test2[i]))**0.5)
    return proba_test

proba_test = blending(proba_test1,proba_test2)
#
# print(proba_test[:3])

#X_train.xgboost.plot_importance


# submission
def submisson(proba_test):
    df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)
    return None

submisson(proba_test)

def testModel(filename):

    def loadDataSet():
        pred = [];
        act = []
        f1 = open(filename)
        #自己建的测试集
        f2 = open('test2.csv')
        next(f1)
        next(f2)
        for line in f1.readlines():
            lineArr = line.strip().split(',')
            pred.append(lineArr[1])

        for line in f2.readlines():
            arr = line.strip().split(',')
            act.append(arr[0])
        print(pred[:10], '@@@@', act[:10])
        return act, pred

    # def logloss(act, pred):
    #   #epsilon = 1e-15
    #   # pred = sp.maximum(epsilon, pred)
    #   # pred = sp.minimum(1-epsilon, pred)
    #   ll = sum(act*sp.log(pred) + 	sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    #   ll = ll * -1.0/len(act)
    #   return ll

    def logloss(act, pred):
        ll = 0
        # epsilon = 1e-15
        # pred = sp.maximum(epsilon, pred)
        # pred = sp.minimum(1-epsilon, pred)
        for i in range(len(act)):
            ll = ll + (
            int(act[i]) * sp.log(float(pred[i])) + sp.subtract(1, int(act[i])) * sp.log(sp.subtract(1, float(pred[i]))))
        ll = ll * -1.0 / len(act)
        return ll

    act, pred = loadDataSet()
    return logloss(act, pred)

logloss = testModel('submission.csv')
print(logloss)

# pd.DataFrame({"columns":list(dfTrain.columns)[1:], "coef":list(lr.coef_.T)})
# import pandas as pd
# print(pd.Series(model.booster().get_fscore()).sort_values(ascending=False))
