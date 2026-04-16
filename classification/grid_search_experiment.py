# -- coding: utf-8 --
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV  # 导入Lasso工具包LassoCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # 分割训练集和验证集
from sklearn.preprocessing import StandardScaler  # 标准化工具包StandardScaler

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    random_state = None  # 固定随机种子
    Lasso = False
    if Lasso == True:
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
    T = False
    if T == True:
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
        # 数据只保留从T检验筛选出的特征数据，重新组合成data
        if not 'label' in columns_index:
            columns_index = ['label'] + columns_index
        MCN_train = MCN_data[columns_index]
        SCN_train = SCN_data[columns_index]
        data = pd.concat([MCN_train, SCN_train])
        data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱
    MUSE = False
    if MUSE == True:
        # 缪斯选择器筛选特征
        # 主要思想是在一个特征下，不同 类别的分布是有明显差异的，如果各个类别都是均匀分布，那这个特征就没有用。
        from kydavra import MUSESelector

        max_columns_num = 60  # 这个值是人工定义值
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
    if Lasso == True:
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
        lassoCV_model = LassoCV(alphas=alpha_range, cv=10, max_iter=100000)
        lassoCV_model.fit(lassoCV_x, lassoCV_y)
        print(lassoCV_model.alpha_)
        coef = pd.Series(lassoCV_model.coef_, index=columnNames)
        print("从原来{}个特征，筛选剩下{}个".format(len(columnNames), sum(coef != 0)))
        print("分别是以下特征")
        print(coef[coef != 0])
        index_ = coef[coef != 0].index
        # pd.DataFrame(data,columns=index_).to_csv('data.csv')
        MCN_data_select = pd.DataFrame(MCN_data, columns=index_)
        MCN_data_select.to_csv('MCN_data_select.csv')
        SCN_data_select = pd.DataFrame(SCN_data, columns=index_)
        SCN_data_select.to_csv('SCN_data_select.csv')
        X = x[index_]
        y = y
    else:
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
        X = data[data.columns[1:]]
        y = data['label']
    standardscaler = StandardScaler()
    X = standardscaler.fit_transform(X)  # 对x进行均值-标准差归一化
    TIMES = 10
    Score = {}
    Cross_score = {}
    # 分类
    NAME = ['svm', 'forest', 'knn', 'bayes', 'MLP', 'adaboost', 'decisiontree', 'logistic', 'SGD', 'xgboost', 'bagging',
            'GB']
    for modelname in NAME[-1:]:
        print('-' * 100)
        print('Model:{}'.format(modelname))
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        print(x_train.shape)
        # PCA
        from sklearn.decomposition import PCA

        model_pca = PCA(n_components=0.6)
        model_pca.fit(x_train)
        x_train = model_pca.transform(x_train)
        x_test = model_pca.transform(x_test)
        pd.DataFrame(x_train).to_csv('x_train.csv')
        print(x_train.shape)
        if modelname == 'forest':
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=30, random_state=random_state)
        elif modelname == 'svm':
            from sklearn import svm

            model = svm.SVC(probability=True, gamma='scale', kernel='linear', C=0.1, cache_size=1, max_iter=30)
            scorer = make_scorer(f1_score)
            parameters = {'max_iter': [i * 10 for i in range(1, 4)], 'C': [i / 10.0 for i in range(1, 5)]}
            grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            grid_obj.fit(x_train, y_train)
            model = grid_obj.best_estimator_
            print(grid_obj.best_params_)
        elif modelname == 'adaboost':
            from sklearn.ensemble import AdaBoostClassifier

            # n_estimators表示要组合的弱分类器个数；
            # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
            model = AdaBoostClassifier(n_estimators=20, learning_rate=0.1, algorithm='SAMME.R')
            scorer = make_scorer(f1_score)
            parameters = {'n_estimators': [10, 20, 30, 40, 50], 'learning_rate': [i / 10.0 for i in range(1, 11)]}
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
            from sklearn.naive_bayes import GaussianNB

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
            # parameters = { 'C': [i / 10.0 for i in range(10)]}
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
        elif modelname == 'bagging':
            from sklearn.ensemble import BaggingClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.linear_model import LogisticRegression

            model = BaggingClassifier(LogisticRegression(solver='liblinear', penalty='l2', max_iter=50, C=0.4),
                                      n_estimators=10,
                                      max_samples=0.5, max_features=0.5)
            # scorer = make_scorer(f1_score)
            # parameters = { 'n_estimators': [10,20,30,40,50],'max_samples': [i/10.0 for i in range(5,8)],
            #                 'max_features': [i/10.0 for i in range(5,8)]}
            # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            # grid_obj.fit(x_train, y_train)
            # model = grid_obj.best_estimator_
            # print(grid_obj.best_params_)
        elif modelname == 'GB':
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,
                                               max_depth=1)
            scorer = make_scorer(f1_score)
            parameters = {'n_estimators': [10, 20, 30, 40], 'learning_rate': [i / 10.0 for i in range(1, 11)]}
            grid_obj = GridSearchCV(model, parameters, scoring=scorer)
            grid_obj.fit(x_train, y_train)
            model = grid_obj.best_estimator_
            print(grid_obj.best_params_)
        else:
            print('No this method')
            raise ('Error')
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        score = round(score, 2)
        Score[modelname] = score
        print("在验证集上的准确率：{}".format(score))
        cross_score = cross_val_score(model, X, y, cv=10)
        mean_score = round(np.mean(cross_score), 2)
        Cross_score[modelname] = mean_score
        print("交叉验证：{}".format(cross_score))
        print("平均：{}".format(mean_score))
        # #绘制混淆矩阵
        # from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
        # predict_label = model.predict(x_test) #预测的标签
        # label = y_test.to_list()  #真实标签
        # print(' Truth :',label)
        # print('Predict:',predict_label.tolist())
        # confusion = confusion_matrix(label, predict_label)#计算混淆矩阵
        # print("验证集一共有{}行特征数据，{}列不同特征,包含MCN:{}例，SCN:{}例".format(len(x_test),x_test.shape,np.sum(label),len(label)-np.sum(label)))
        # print("混淆矩阵为：\n{}".format(confusion))
        # print("\n计算各项指标：")
        # print(classification_report(label, predict_label))
        # 绘制ROC曲线,方法1
        # from sklearn.metrics import roc_curve, roc_auc_score, auc
        # kind = {'MCN': 1, "SCN": 0}
        # label = y_test.to_list()  # 真实标签
        # y_predict = model.predict_proba(x_test)  # 得到标签0和1对应的概率
        # fpr, tpr, threshold = roc_curve(label, y_predict[:, kind['SCN']], pos_label=kind['SCN'])
        # roc_auc = auc(fpr, tpr)  # 计算auc的
        # fpr1, tpr1, threshold = roc_curve(label, y_predict[:, kind['MCN']], pos_label=kind['MCN'])
        # roc_auc1 = auc(fpr1, tpr1)  # 计算auc的
        # plt.figure(figsize=(6, 5))
        # plt.plot(fpr, tpr, marker='o', markersize=5, label='SCN')
        # plt.plot(fpr1, tpr1, marker='*', markersize=5, label='MCN')
        # plt.title("SCN AUC:{:.2f}, MCN AUC:{:.2f}".format(roc_auc, roc_auc1))
        # plt.xlabel('FPR')
        # plt.ylabel('TPR')
        # plt.legend(loc=4)
        # # 绘制ROC方法2,两行代码
        # from sklearn.metrics import plot_roc_curve
        # ax1 = plot_roc_curve(model, x_test, y_test, name='SCN', pos_label=0)
        # plot_roc_curve(model, x_test, y_test, ax=ax1.ax_, name='MCN', pos_label=1)
        # plt.show()
        # with open('./model/'+modelname+'.txt', 'a+', encoding='utf-8') as f:
        #     f.write(str(score)+'\t'+str(np.mean(mean_score))+'\n')
        #     f.close()
    print('Score:', Score)
    print("Cross_score:", Cross_score)
    # print('Score:')
    # for s in Score:
    #     print(s,Score[s])
    # print("Cross_score:")
    # for s in Cross_score:
    #     print(s,Cross_score[s])
