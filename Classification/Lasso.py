# -- coding: utf-8 --
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV  # 导入Lasso工具包LassoCV
from sklearn.preprocessing import StandardScaler  # 标准化工具包StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # 分割训练集和验证集
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
random_state = None  # 固定随机种子

MCN_data = pd.read_csv('./MCN.csv')
SCN_data = pd.read_csv('./SCN.csv')
MCN_data.insert(0, 'label', 1)  # 插入标签
SCN_data.insert(0, 'label', 0)  # 插入标签
MCN_data = MCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
SCN_data = SCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
# 因为有些特征是字符串，直接删掉
cols = [x for i, x in enumerate(MCN_data.columns) if type(MCN_data.iat[1, i]) == str]
MCN_data = MCN_data.drop(cols, axis=1)
cols = [x for i, x in enumerate(SCN_data.columns) if type(SCN_data.iat[1, i]) == str]
SCN_data = SCN_data.drop(cols, axis=1)
# 查看总体数据情况
data = pd.concat([MCN_data, SCN_data])
data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱
print("一共有{}行特征数据".format(len(data)))
print("一共有{}列不同特征".format(data.shape[1]))
# 再把特征值数据和标签数据分开
x = data[data.columns[1:]]
y = data['label']
# 先保存X的列名
columnNames = x.columns
lassoCV_x = x.astype(np.float32)  # 把x数据转换成np.float格式
lassoCV_y = y
standardscaler = StandardScaler()
lassoCV_x = standardscaler.fit_transform(lassoCV_x)  # 对x进行均值-标准差归一化
lassoCV_x = pd.DataFrame(lassoCV_x, columns=columnNames)  # 转 DataFrame 格式
# alpha_range = np.logspace(-3,-2,50,base=5)
alpha_range = np.logspace(-3, 1, 50)
lassoCV_model = LassoCV(alphas=alpha_range, cv=10, max_iter=10000)
lassoCV_model.fit(lassoCV_x, lassoCV_y)
print(lassoCV_model.alpha_)
coef = pd.Series(lassoCV_model.coef_, index=columnNames)
print("从原来{}个特征，筛选剩下{}个".format(len(columnNames), sum(coef != 0)))
print("分别是以下特征")
print(coef[coef != 0])
index_ = coef[coef != 0].index
# coefs = lassoCV_model.path(lassoCV_x,lassoCV_y, alphas=alpha_range,max_iter=10000)[1].T
# plt.figure(figsize=(8,6))
# plt.semilogx(lassoCV_model.alphas_,coefs,'-')
# plt.axvline(lassoCV_model.alpha_,color = 'black',ls = '--')
# plt.xlabel('Lamda')
# plt.ylabel('Coefficient')
# plt.show()
# pd.DataFrame(data,columns=index_).to_csv('data.csv')
MCN_data = pd.read_csv('./MCN.csv')
SCN_data = pd.read_csv('./SCN.csv')
MCN_data_select = pd.DataFrame(MCN_data, columns=index_)
MCN_data_select.to_csv('MCN_data_select.csv')
SCN_data_select = pd.DataFrame(SCN_data, columns=index_)
SCN_data_select.to_csv('SCN_data_select.csv')
