# -- coding: utf-8 --
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from common import build_features_and_labels, load_labeled_data, merge_and_shuffle
from compute_metric import calculate_metric


T = time.time()
warnings.filterwarnings('ignore')
random_state = None

print('-' * 100)
print('Read select data')

mcn_data, scn_data = load_labeled_data('MCN_data_select.csv', 'SCN_data_select.csv', random_state=random_state)
data = merge_and_shuffle(mcn_data, scn_data, random_state=random_state)
print('一共{}行特征数据'.format(len(data)))
print('一共{}列不同特征'.format(data.shape[1]))

column_names = data.columns[2:]
X, y = build_features_and_labels(data, feature_start_col=2, scale=True)
X = pd.DataFrame(X, columns=column_names)
print(X.shape)

lsvc = LinearSVC(C=0.1, penalty='l2', dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X = model.transform(X)

selected_columns = column_names[model.get_support()]
print(f'Features selected by SelectFromModel: {len(selected_columns)}')
print(f'Features selected by SelectFromModel: {selected_columns}')

mcn_raw = pd.read_csv('./MCN.csv')
scn_raw = pd.read_csv('./SCN.csv')
pd.DataFrame(mcn_raw, columns=selected_columns).to_csv('MCN_data_select2.csv')
pd.DataFrame(scn_raw, columns=selected_columns).to_csv('SCN_data_select2.csv')

print(X.shape)
S = []
for _ in range(1):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn import svm

    clf = svm.SVC(probability=True, gamma='scale', kernel='linear', C=0.1, cache_size=1, max_iter=30)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    S.append(score)

    predict_label = clf.predict(x_test)
    label = y_test.to_list()
    calculate_metric(label, predict_label)

print('Mean accuracy: %0.2f ' % (np.mean(S)))
print('Run time {} s'.format(time.time() - T))
