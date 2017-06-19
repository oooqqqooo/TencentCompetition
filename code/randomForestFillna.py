from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

df = pd.read_csv('E:\\数据集\\pre\\数据预处理\\v5.csv')

### 使用 RandomForestClassifier 填补缺失的属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    num_df = df[['countNum','age','gender','education','marriageStatus','haveBaby','hometown','residence']]

    # 用户分成已知count和未知count两部分
    known_num = num_df[num_df.countNum.notnull()].as_matrix()
    unknown_num = num_df[num_df.countNum.isnull()].as_matrix()

    # y即目标Num
    y = known_num[:, 0]

    # X即特征属性值
    X = known_num[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedNum = rfr.predict(unknown_num[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.actionNum.isnull()), 'countNum' ] = predictedNum

    return df, rfr

dd = set_missing_ages(df)
np.savetxt('E:\\数据集\\pre\\数据预处理\\v6.csv',dd, fmt="%s", delimiter=',')