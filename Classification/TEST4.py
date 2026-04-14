# -- coding: utf-8 --
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # 标准化工具包StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # 分割训练集和验证集
import matplotlib.pyplot as plt
from compute_metric import calculate_metric
import warnings
from tqdm import tqdm
import time

T = time.time()
warnings.filterwarnings("ignore")
random_state = None  # 固定随机种子
NAME = ['SVM', 'RF', 'KNN', 'Bayes', 'MLP', 'Adaboost', 'Decisiontree', 'Logistic', 'SGD', 'XGBoost']
Score = {}
Cross_score = {}
metric = ['auc', 'acc', 'sen', 'spe', 'pre', 'f1']
# 分类
for modelname in NAME[:-1]:
    print('-' * 100)
    print('Model:{}'.format(modelname))
    s = {'auc': [], 'acc': [], 'sen': [], 'spe': [], 'pre': [], 'f1': []}
    S = {'auc': [], 'acc': [], 'sen': [], 'spe': [], 'pre': [], 'f1': []}
    FPR = []
    TPR = []
    print("Read select data")
    MCN_data = pd.read_csv('MCN_data_select2.csv')
    SCN_data = pd.read_csv('SCN_data_select2.csv')
    MCN_data.insert(0, 'label', 1)  # 插入标签
    SCN_data.insert(0, 'label', 0)  # 插入标签
    MCN_data = MCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
    SCN_data = SCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
    data = pd.concat([MCN_data, SCN_data])
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 10000)
    fig, ax = plt.subplots()
    for i in tqdm(range(1000)):
        data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱
        X = data[data.columns[2:]]
        y = data['label']
        standardscaler = StandardScaler()
        X = standardscaler.fit_transform(X)  # 对x进行均值-标准差归一化
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        from Model import Model

        model = Model(modelname, x_train, y_train)
        model.fit(x_train, y_train)
        predict_label = model.predict(x_test)  # 预测的标签
        label = y_test.to_list()  # 真实标签
        auc, acc, sen, spe, pre, f1 = calculate_metric(label, predict_label)
        for m, a in zip(metric, [auc, acc, sen, spe, pre, f1]):
            s[m].append(a)
        from sklearn.metrics import RocCurveDisplay, roc_curve, auc

        fpr, tpr, thresholds = roc_curve(label, predict_label, pos_label=1)
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    # print(s)
    for m in metric:
        mean_score = round(np.mean(s[m]), 2)
        std = round(np.std(s[m]), 2)
        S[m] = str(mean_score) + '±' + str(std)
    Score[modelname] = S
    from Roc_plot import Mean_roc_plot

    Mean_roc_plot(ax, tprs, aucs, mean_fpr, modelname)
    print(S)
print(Score)
pd.DataFrame(Score).T.to_csv('metric.csv', encoding='utf-8')
print('Run time {} s'.format(time.time() - T))
