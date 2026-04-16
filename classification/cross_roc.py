# -- coding: utf-8 --
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold

from classification.common import build_features_and_labels, load_labeled_data, merge_and_shuffle


random_state = None
print('-' * 100)
print('Read select data')

mcn_data, scn_data = load_labeled_data('MCN_data_select2.csv', 'SCN_data_select2.csv', random_state=random_state)
data = merge_and_shuffle(mcn_data, scn_data, random_state=random_state)
X, y = build_features_and_labels(data, feature_start_col=2, scale=True)
y = np.asarray(y)

cv = StratifiedKFold(n_splits=10, shuffle=True)
classifier = svm.SVC(probability=True, gamma='scale', kernel='linear', C=0.1, cache_size=1, max_iter=30)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 10000)
fig, ax = plt.subplots()

for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    print(classifier.score(X[test], y[test]))
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name='ROC fold {}'.format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print(mean_auc)
print(np.mean(aucs))
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color='b',
    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color='grey',
    alpha=0.2,
    label=r'$\pm$ 1 std. dev.',
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title='Receiver operating characteristic example',
)
ax.legend(loc='lower right')
plt.show()
