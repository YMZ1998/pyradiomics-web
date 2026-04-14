# -- coding: utf-8 --
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV  # 导入Lasso工具包LassoCV
from sklearn.preprocessing import StandardScaler  # 标准化工具包StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # 分割训练集和验证集
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
import warnings
import time

T = time.time()
warnings.filterwarnings("ignore")
random_state = None  # 固定随机种子
NAME = ['svm', 'forest', 'knn', 'bayes', 'MLP', 'adaboost', 'decisiontree', 'logistic', 'SGD', 'xgboost']
Score = {}
Cross_score = {}
# 分类
for modelname in NAME[:]:
    print('-' * 100)
    print('Model:{}'.format(modelname))
    s = []
    c = []
    print("Read select data")
    MCN_data = pd.read_csv('MCN_data_select2.csv')
    SCN_data = pd.read_csv('SCN_data_select2.csv')
    MCN_data.insert(0, 'label', 1)  # 插入标签
    SCN_data.insert(0, 'label', 0)  # 插入标签
    MCN_data = MCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
    SCN_data = SCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
    data = pd.concat([MCN_data, SCN_data])
    for i in range(1000):
        data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱
        # print("一共有{}行特征数据".format(len(data)))
        # print("一共有{}列不同特征".format(data.shape[1]))
        X = data[data.columns[2:]]
        y = data['label']
        standardscaler = StandardScaler()
        X = standardscaler.fit_transform(X)  # 对x进行均值-标准差归一化
        # feature
        from sklearn.svm import LinearSVC
        from sklearn.feature_selection import SelectFromModel

        # lsvc = LinearSVC(C=0.1, penalty="l2", dual=False).fit(X, y)
        # model = SelectFromModel(lsvc, prefit=True)
        # X = model.transform(X)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # PCA
        # from sklearn.decomposition import PCA
        # model_pca = PCA(n_components = 0.5)
        # model_pca.fit(x_train)
        # x_train = model_pca.transform(x_train)
        # x_test= model_pca.transform(x_test)
        # pd.DataFrame(x_train).to_csv('x_train.csv')
        # print(x_train.shape)
        if modelname == 'forest':
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=30, random_state=random_state)
        elif modelname == 'svm':
            from sklearn import svm

            model = svm.SVC(probability=True, gamma='scale', kernel='linear', C=0.1, cache_size=1, max_iter=30)
            # scorer = make_scorer(f1_score)
            # parameters = {'max_iter': [i*10 for i in range(1,4)],'C': [i/10.0 for i in range(1,5)]}
            # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            # grid_obj.fit(x_train, y_train)
            # model = grid_obj.best_estimator_
            # print(grid_obj.best_params_)
        elif modelname == 'adaboost':
            from sklearn.ensemble import AdaBoostClassifier

            # n_estimators表示要组合的弱分类器个数；
            # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
            model = AdaBoostClassifier(n_estimators=20, learning_rate=0.1, algorithm='SAMME.R')
            scorer = make_scorer(f1_score)
            parameters = {'n_estimators': [10, 20, 30], 'learning_rate': [0.1, 0.5, 1.0]}
            grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            grid_obj.fit(x_train, y_train)
            model = grid_obj.best_estimator_
            print(grid_obj.best_params_)
        elif modelname == 'decisiontree':
            from sklearn.tree import DecisionTreeClassifier

            # criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）
            model = DecisionTreeClassifier(criterion='entropy', max_depth=2,
                                           class_weight='balanced', min_samples_leaf=2, min_samples_split=2)
            scorer = make_scorer(f1_score)
            parameters = {'max_depth': [2, 3, 4], 'min_samples_leaf': [2, 3, 4], 'min_samples_split': [2, 3, 4]}
            grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            grid_obj.fit(x_train, y_train)
            model = grid_obj.best_estimator_
            print(grid_obj.best_params_)
        elif modelname == 'bayes':
            from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB

            model = GaussianNB()
        elif modelname == 'MLP':
            from sklearn.neural_network import MLPClassifier

            model = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(12, 3), max_iter=100,
                                  random_state=1, learning_rate_init=1e-5)
            # scorer = make_scorer(f1_score)
            # parameters = {'hidden_layer_sizes':[(i,3) for i in range(10,15)]}
            # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            # grid_obj.fit(x_train, y_train)
            # model = grid_obj.best_estimator_
            # print(grid_obj.best_params_)
        elif modelname == 'knn':
            from sklearn.neighbors import KNeighborsClassifier

            model = KNeighborsClassifier(n_neighbors=3, leaf_size=1)
            scorer = make_scorer(f1_score)
            parameters = {'n_neighbors': [1, 3], 'p': [i for i in range(1, 6)]}
            grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            grid_obj.fit(x_train, y_train)
            model = grid_obj.best_estimator_
            print(grid_obj.best_params_)
        elif modelname == 'logistic':
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(solver='liblinear', penalty='l2', max_iter=50, C=0.4)
            # scorer = make_scorer(f1_score)
            # parameters = { 'max_iter': [i *10 for i in range(10)],'C': [i / 10.0 for i in range(10)]}
            # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            # grid_obj.fit(x_train, y_train)
            # model = grid_obj.best_estimator_
            # print(grid_obj.best_params_)
        elif modelname == 'SGD':
            from sklearn.linear_model import SGDClassifier

            model = SGDClassifier(loss="log", penalty="l2")
        elif modelname == 'xgboost':
            import xgboost as xgb

            model = xgb.XGBClassifier(
                learning_rate=0.1,
                n_estimators=500,
                max_depth=3,
                min_child_weight=1,
                gamma=0,
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=0.005,
                objective='binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=27)
            # scorer = make_scorer(f1_score)
            # parameters = {'learning_rate':[0.005,0.01,0.1]
            #               }
            # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            # grid_obj.fit(x_train, y_train)
            # model = grid_obj.best_estimator_
            # print(grid_obj.best_params_)
        else:
            print('No this method')
            raise ('Error')
        cross_score = cross_val_score(model, X, y, cv=10)
        mean = round(np.mean(cross_score), 4)
        c.append(mean)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        s.append(score)
        # 绘制混淆矩阵
        # from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
        #
        # predict_label = model.predict(x_test)  # 预测的标签
        # label = y_test.to_list()  # 真实标签
        # print(' Truth :', label)
        # print('Predict:', predict_label.tolist())
        # confusion = confusion_matrix(label, predict_label)  # 计算混淆矩阵
        # print("验证集一共有{}行特征数据，{}列不同特征,包含MCN:{}例，SCN:{}例".format(len(x_test), x_test.shape[1], np.sum(label),
        #                                                        len(label) - np.sum(label)))
        # print("混淆矩阵为：\n{}".format(confusion))
        # print("\n计算各项指标：")
        # print(classification_report(label, predict_label))
    print(c)
    print(s)
    mean_score = round(np.mean(s) * 100, 2)
    std = round(np.std(s * 100), 2)
    Score[modelname] = str(mean_score) + '±' + str(std)
    print("mean_score：{}".format(mean_score))
    mean_cross_score = round(np.mean(c) * 100, 2)
    cross_std = round(np.std(c * 100), 2)
    Cross_score[modelname] = str(mean_cross_score) + '±' + str(cross_std)
    print("mean_cross_score：{}".format(mean_cross_score))
print('Score:', Score)
print("Cross_score:", Cross_score)
print('Run time {} s'.format(time.time() - T))
