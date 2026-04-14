# -- coding: utf-8 --
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV  # 导入Lasso工具包LassoCV
from sklearn.preprocessing import StandardScaler  # 标准化工具包StandardScaler

random_state = 100  # 固定随机种子
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

from scipy.stats import levene, ttest_ind

counts = 0
columns_index = []
for column_name in MCN_data.columns[1:]:
    if levene(MCN_data[column_name], SCN_data[column_name])[1] > 0.05:
        if ttest_ind(MCN_data[column_name], SCN_data[column_name], equal_var=True)[1] < 0.05:
            columns_index.append(column_name)
    else:
        if ttest_ind(MCN_data[column_name], SCN_data[column_name], equal_var=False)[1] < 0.05:
            columns_index.append(column_name)
print("筛选后剩下的特征数：{}个".format(len(columns_index)))
from kydavra import MUSESelector

# 数据只保留从T检验筛选出的特征数据，重新组合成data
if not 'label' in columns_index:
    columns_index = ['label'] + columns_index
MCN_train = MCN_data[columns_index]
SCN_train = SCN_data[columns_index]
data = pd.concat([MCN_train, SCN_train])
data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱
# 缪斯选择器筛选特征
# 主要思想是在一个特征下，不同 类别的分布是有明显差异的，如果各个类别都是均匀分布，那这个特征就没有用。
max_columns_num = 20  # 这个值是人工定义值
selector = MUSESelector(num_features=max_columns_num)
# selector = LassoSelector()
columns_index = selector.select(data, 'label')
print("筛选后剩下的特征数：{}个".format(len(columns_index)))
# Lasso
# 数据只保留从T检验筛选出的特征数据，重新组合成data
if not 'label' in columns_index:
    columns_index = ['label'] + columns_index
MCN_train = MCN_data[columns_index]
SCN_train = SCN_data[columns_index]

data = pd.concat([MCN_train, SCN_train])
data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱

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

# 形成5为底的指数函数
# 5**（-3） ~  5**（-2）
# alpha_range = np.logspace(-3,-2,50,base=5)
alpha_range = np.logspace(-3, 2, 50)
# alpha_range在这个参数范围里挑出aplpha进行训练，cv是把数据集分5分，进行交叉验证，max_iter是训练1000轮
lassoCV_model = LassoCV(alphas=alpha_range, cv=5, max_iter=100000)
# 进行训练
lassoCV_model.fit(lassoCV_x, lassoCV_y)
# 打印训练找出来的入值
print(lassoCV_model.alpha_)

coef = pd.Series(lassoCV_model.coef_, index=columnNames)
print("从原来{}个特征，筛选剩下{}个".format(len(columnNames), sum(coef != 0)))
print("分别是以下特征")
print(coef[coef != 0])
# 分类
NAME = ['svm', 'forest', 'knn', 'bayes', 'MLP', 'adaboost', 'decisiontree', 'logistic', 'SGD']
modelname = NAME[0]
print('Model:{}'.format(modelname))
from sklearn.model_selection import train_test_split, cross_val_score  # 分割训练集和验证集

index_ = coef[coef != 0].index
pd.DataFrame(data, columns=index_).to_csv('data.csv')
# MCN_data_select=pd.DataFrame(MCN_data,columns=index_)
# MCN_data_select.to_csv('MCN_data_select.csv')
# SCN_data_select=pd.DataFrame(SCN_data,columns=index_)
# SCN_data_select.to_csv('SCN_data_select.csv')
# print('特征索引：',index_)
X = x[index_]
y = y
standardscaler = StandardScaler()
X = standardscaler.fit_transform(X)  # 对x进行均值-标准差归一化
TIMES = 10
Score = []
for i in range(1, TIMES + 1):
    print('第{}次：'.format(i))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # PCA
    from sklearn.decomposition import PCA

    model_pca = PCA(n_components=0.5)
    model_pca.fit(x_train)
    X_train = model_pca.transform(x_train)
    X_test = model_pca.transform(x_test)

    if modelname == 'forest':
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=30, random_state=random_state).fit(x_train, y_train)
    elif modelname == 'svm':
        from sklearn import svm

        model = svm.SVC(kernel='rbf', gamma='auto', probability=True).fit(x_train, y_train)
    elif modelname == 'adaboost':
        from sklearn.ensemble import AdaBoostClassifier

        # n_estimators表示要组合的弱分类器个数；
        # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
        model = AdaBoostClassifier(n_estimators=10, algorithm='SAMME.R')
        model.fit(x_train, y_train)
    elif modelname == 'decisiontree':
        from sklearn.tree import DecisionTreeClassifier

        # criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(x_train, y_train)
    elif modelname == 'bayes':
        from sklearn.naive_bayes import GaussianNB

        model = GaussianNB()
        model.fit(x_train, y_train)
    elif modelname == 'MLP':
        from sklearn.neural_network import MLPClassifier

        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1)
        model.fit(x_train, y_train)
    elif modelname == 'knn':
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier(n_neighbors=4)
        model.fit(x_train, y_train)
    elif modelname == 'logistic':
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(solver='liblinear', penalty='l2')
        model.fit(x_train, y_train)
    elif modelname == 'SGD':
        from sklearn.linear_model import SGDClassifier

        model = SGDClassifier(loss="log", penalty="l2")
        model.fit(x_train, y_train)
    else:
        print('No this method')
        raise ('Error')
    score = model.score(x_test, y_test)
    Score.append(score)
    print("在验证集上的准确率：{}".format(score))
    cross_score = cross_val_score(model, X, y, cv=10)
    mean_score = round(np.mean(cross_score), 2)
    print("交叉验证：{}".format(cross_score))
    print("平均：{}".format(mean_score))

    # 绘制混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report

    predict_label = model.predict(x_test)  # 预测的标签
    label = y_test.to_list()  # 真实标签
    print(' Truth :', label)
    print('Predict:', predict_label.tolist())
    confusion = confusion_matrix(label, predict_label)  # 计算混淆矩阵
    print("验证集一共有{}行特征数据，{}列不同特征,包含MCN:{}例，SCN:{}例".format(len(x_test), x_test.shape[1],
                                                                               np.sum(label),
                                                                               len(label) - np.sum(label)))
    print("混淆矩阵为：\n{}".format(confusion))
    print("\n计算各项指标：")
    print(classification_report(label, predict_label))

    # with open('./model/'+modelname+'.txt', 'a+', encoding='utf-8') as f:
    #     f.write(str(score)+'\t'+str(np.mean(mean_score))+'\n')
    #     f.close()
print('Score:', Score)
print("Mean:", np.mean(Score))
print("Cross_score：{}".format(mean_score))
with open('./model/' + modelname + '.txt', 'a+', encoding='utf-8') as f:
    f.write(str(round(np.mean(Score), 2)) + '\n')
    f.close()
print('Model:{}'.format(modelname))
