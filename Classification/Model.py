# -- coding: utf-8 --
"""Model factory used by legacy training scripts."""

random_state = None
NAME = ['SVM', 'RF', 'KNN', 'Bayes', 'MLP', 'Adaboost', 'Decisiontree', 'Logistic', 'SGD', 'XGBoost']


def _build_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=30, random_state=random_state)


def _build_svm():
    from sklearn import svm
    return svm.SVC(probability=True, gamma='scale', kernel='linear', C=0.1, cache_size=100, max_iter=-1)


def _build_adaboost():
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(n_estimators=20, learning_rate=0.1, algorithm='SAMME.R')


def _build_decisiontree():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(
        criterion='entropy',
        max_depth=2,
        class_weight='balanced',
        min_samples_leaf=2,
        min_samples_split=2,
    )


def _build_bayes():
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


def _build_mlp():
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(12, 3), max_iter=100, learning_rate_init=1e-5)


def _build_knn():
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=3, leaf_size=1)


def _build_logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(solver='liblinear', penalty='l2', max_iter=50, C=0.4, n_jobs=-1)


def _build_sgd():
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(loss='log', penalty='l2')


def _build_xgboost():
    import xgboost as xgb
    return xgb.XGBClassifier(
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
        seed=27,
        eval_metric=['logloss', 'auc', 'error'],
    )


_MODEL_BUILDERS = {
    'RF': _build_rf,
    'SVM': _build_svm,
    'Adaboost': _build_adaboost,
    'Decisiontree': _build_decisiontree,
    'Bayes': _build_bayes,
    'MLP': _build_mlp,
    'KNN': _build_knn,
    'Logistic': _build_logistic,
    'SGD': _build_sgd,
    'XGBoost': _build_xgboost,
}


def Model(modelname, x_train, y_train):
    """Keep legacy function signature for compatibility."""
    _ = (x_train, y_train)
    try:
        return _MODEL_BUILDERS[modelname]()
    except KeyError:
        print('No this method')
        raise ValueError('Unsupported model: {}'.format(modelname))
