# -- coding: utf-8 --
import time
import warnings

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier

from common import build_features_and_labels, load_labeled_data, merge_and_shuffle


T = time.time()
warnings.filterwarnings('ignore')
random_state = None

print('-' * 100)
print('Read select data')

mcn_data, scn_data = load_labeled_data('MCN_data_select2.csv', 'SCN_data_select2.csv', random_state=random_state)
data = merge_and_shuffle(mcn_data, scn_data, random_state=random_state)
X, y = build_features_and_labels(data, feature_start_col=2, scale=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(x_train.shape)

clf1 = LogisticRegression(solver='liblinear', penalty='l2', max_iter=50, C=0.4)

from sklearn import svm

clf2 = svm.SVC(probability=True, gamma='scale', kernel='linear', C=0.1, cache_size=1, max_iter=30)
clf3 = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(12, 3), max_iter=100, random_state=1, learning_rate_init=1e-5)

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'SVM', 'MLP', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    print('Accuracy: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label))

print('Run time {} s'.format(time.time() - T))
