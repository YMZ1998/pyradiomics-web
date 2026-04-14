# -- coding: utf-8 --
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # 分割训练集和验证集
from sklearn.metrics import make_scorer, f1_score
random_state=None
NAME=['SVM','RF','KNN','Bayes','MLP','Adaboost','Decisiontree','Logistic','SGD','XGBoost']
def Model(modelname,x_train,y_train):
    if modelname == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=30, random_state=random_state)
    elif modelname == 'SVM':
        from sklearn import svm
        model = svm.SVC(probability=True, gamma='scale', kernel='linear', C=0.1, cache_size=100, max_iter=-1)
        # scorer = make_scorer(f1_score)
        # parameters = {'max_iter': [i*10 for i in range(1,4)],'C': [i/10.0 for i in range(1,5)]}
        # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
        # grid_obj.fit(x_train, y_train)
        # model = grid_obj.best_estimator_
        # print(grid_obj.best_params_)
    elif modelname == 'Adaboost':
        from sklearn.ensemble import AdaBoostClassifier
        # n_estimators表示要组合的弱分类器个数；
        # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
        model = AdaBoostClassifier(n_estimators=20, learning_rate=0.1, algorithm='SAMME.R')
        # scorer = make_scorer(f1_score)
        # parameters = {'n_estimators': [10, 20, 30], 'learning_rate': [0.1, 0.5, 1.0]}
        # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
        # grid_obj.fit(x_train, y_train)
        # model = grid_obj.best_estimator_
        # print(grid_obj.best_params_)
    elif modelname == 'Decisiontree':
        from sklearn.tree import DecisionTreeClassifier
        # criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）
        model = DecisionTreeClassifier(criterion='entropy', max_depth=2,
                                       class_weight='balanced', min_samples_leaf=2, min_samples_split=2)
        # scorer = make_scorer(f1_score)
        # parameters = {'max_depth': [2, 3, 4], 'min_samples_leaf': [2, 3, 4], 'min_samples_split': [2, 3, 4]}
        # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
        # grid_obj.fit(x_train, y_train)
        # model = grid_obj.best_estimator_
        # print(grid_obj.best_params_)
    elif modelname == 'Bayes':
        from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
        model = GaussianNB()
    elif modelname == 'MLP':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(12, 3), max_iter=100,
                               learning_rate_init=1e-5)
        # scorer = make_scorer(f1_score)
        # parameters = {'hidden_layer_sizes':[(i,3) for i in range(10,15)]}
        # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
        # grid_obj.fit(x_train, y_train)
        # model = grid_obj.best_estimator_
        # print(grid_obj.best_params_)
    elif modelname == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=3, leaf_size=1)
        # scorer = make_scorer(f1_score)
        # parameters = {'n_neighbors': [1, 3], 'p': [i for i in range(1, 6)]}
        # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
        # grid_obj.fit(x_train, y_train)
        # model = grid_obj.best_estimator_
        # print(grid_obj.best_params_)
    elif modelname == 'Logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver='liblinear', penalty='l2', max_iter=50, C=0.4,n_jobs=-1)
        # scorer = make_scorer(f1_score)
        # parameters = { 'max_iter': [i *10 for i in range(10)],'C': [i / 10.0 for i in range(10)]}
        # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
        # grid_obj.fit(x_train, y_train)
        # model = grid_obj.best_estimator_
        # print(grid_obj.best_params_)
    elif modelname == 'SGD':
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss="log", penalty="l2")
        # scorer = make_scorer(f1_score)
        # parameters = { 'max_iter': [i *10 for i in range(10)],'C': [i / 10.0 for i in range(10)]}
        # grid_obj = GridSearchCV(model, parameters, scoring=scorer)
        # grid_obj.fit(x_train, y_train)
        # model = grid_obj.best_estimator_
    elif modelname == 'XGBoost':
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
            seed=27,eval_metric=['logloss','auc','error'])
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
    return model