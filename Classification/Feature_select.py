# -- coding: utf-8 --
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV  # 导入Lasso工具包LassoCV
from sklearn.preprocessing import StandardScaler  # 标准化工具包StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # 分割训练集和验证集
from sklearn.metrics import make_scorer, f1_score
import warnings
import time

T = time.time()
warnings.filterwarnings("ignore")
random_state = None  # 固定随机种子
print('-' * 100)
print("Read select data")
MCN_data = pd.read_csv('MCN_data_select.csv')
SCN_data = pd.read_csv('SCN_data_select.csv')
MCN_data.insert(0, 'label', 1)  # 插入标签
SCN_data.insert(0, 'label', 0)  # 插入标签
MCN_data = MCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
SCN_data = SCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
data = pd.concat([MCN_data, SCN_data])
data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱
print("一共有{}行特征数据".format(len(data)))
print("一共有{}列不同特征".format(data.shape[1]))

columnNames = data.columns[2:]
X = data[data.columns[2:]]
y = data['label']
standardscaler = StandardScaler()
X = standardscaler.fit_transform(X)  # 对x进行均值-标准差归一化
X = pd.DataFrame(X, columns=columnNames)
print(X.shape)
# https://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection
# method1
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=0.1, penalty="l2", dual=False).fit(X, y)
# from sklearn.feature_selection import SequentialFeatureSelector
# model = SequentialFeatureSelector(
#     lsvc, n_features_to_select=10, direction="forward").fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X = model.transform(X)
# print(columnNames)
print(f"Features selected by SelectFromModel: {len(columnNames[model.get_support()])}")
print(f"Features selected by SelectFromModel: {columnNames[model.get_support()]}")
index_ = columnNames[model.get_support()]
MCN_data = pd.read_csv('./MCN.csv')
SCN_data = pd.read_csv('./SCN.csv')
MCN_data_select = pd.DataFrame(MCN_data, columns=index_)
MCN_data_select.to_csv('MCN_data_select2.csv')
SCN_data_select = pd.DataFrame(SCN_data, columns=index_)
SCN_data_select.to_csv('SCN_data_select2.csv')
# method3
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectFromModel
# clf = ExtraTreesClassifier(criterion='entropy',n_estimators=30)
# clf = clf.fit(X, y)
# model = SelectFromModel(clf, prefit=True)
# X = model.transform(X)
print(X.shape)
S = []
for i in range(1):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # lsvc = LinearSVC(C=0.1, penalty="l2", dual=False).fit(x_train, y_train)
    # model = SelectFromModel(lsvc, prefit=True)
    # print(x_train.shape)
    # x_train = model.transform(x_train)
    # x_test = model.transform(x_test)
    # print(x_train.shape)
    from sklearn import svm

    clf = svm.SVC(probability=True, gamma='scale', kernel='linear', C=0.1, cache_size=1, max_iter=30)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    S.append(score)
    # print("Accuracy: %0.2f " % (score))
    from compute_metric import calculate_metric

    predict_label = clf.predict(x_test)  # 预测的标签
    label = y_test.to_list()  # 真实标签
    calculate_metric(label, predict_label)
print("Mean accuracy: %0.2f " % (np.mean(S)))
