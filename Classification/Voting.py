# -- coding: utf-8 --
# -- coding: utf-8 --
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
MCN_data = pd.read_csv('MCN_data_select2.csv')
SCN_data = pd.read_csv('SCN_data_select2.csv')
MCN_data.insert(0, 'label', 1)  # 插入标签
SCN_data.insert(0, 'label', 0)  # 插入标签
MCN_data = MCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
SCN_data = SCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
data = pd.concat([MCN_data, SCN_data])
data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱
X = data[data.columns[2:]]
y = data['label']
standardscaler = StandardScaler()
X = standardscaler.fit_transform(X)  # 对x进行均值-标准差归一化

# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import SelectFromModel
# lsvc = LinearSVC(C=0.1, penalty="l2", dual=False).fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
# X = model.transform(X)
# print(X.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(x_train.shape)
# lsvc = LinearSVC(C=0.1, penalty="l2", dual=False).fit(x_train,y_train)
# model = SelectFromModel(lsvc, prefit=True)
# x_train = model.transform(x_train)
# x_test = model.transform(x_test)
# X = model.transform(X)
# print(x_train.shape)
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(solver='liblinear', penalty='l2', max_iter=50, C=0.4)
# clf2 = RandomForestClassifier(n_estimators=30,random_state=random_state)
from sklearn import svm

clf2 = svm.SVC(probability=True, gamma='scale', kernel='linear', C=0.1, cache_size=1, max_iter=30)
from sklearn.neural_network import MLPClassifier

clf3 = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(12, 3), max_iter=100,
                     random_state=1, learning_rate_init=1e-5)

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='hard')
# eclf= VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
#                         voting='soft', weights=[2, 2, 1])
for clf, label in zip([clf1, clf2, clf3, eclf],
                      ['Logistic Regression', 'SVM', 'MLP', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    # clf.fit(x_train, y_train)
    # score=clf.score(x_test,y_test)
    # print("Accuracy: %0.2f [%s]" % (score, label))

print('Run time {} s'.format(time.time() - T))
