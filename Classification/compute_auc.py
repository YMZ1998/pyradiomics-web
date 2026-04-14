from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, \
    classification_report, roc_auc_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve

# MCN 1 SCN 0   51  56
labels = ['SCN', 'MCN']
MCN_error = 5
SCN_error = 5

pred = []
truth = []
# 预测正确
for i in range(51 - MCN_error):
    pred.append(1)
# 错误
for i in range(MCN_error):
    pred.append(0)
# 正确
for i in range(56 - SCN_error):
    pred.append(0)
# 错误
for i in range(SCN_error):
    pred.append(1)
# Truth
for i in range(51):
    truth.append(1)
for i in range(56):
    truth.append(0)
from sklearn.utils import shuffle

# pred=shuffle(pred,random_state=2)
# truth=shuffle(truth,random_state=2)
print(pred)
print(truth)
from compute_metric import calculate_metric

calculate_metric(truth, pred, verbose=1)
print(classification_report(truth, pred, target_names=labels))
# print("Precision: "+ str(precision_score(truth, pred, average='weighted')))
# print("Recall (Sensitivity): "+ str(recall_score(truth, pred, average='weighted')))
# print("Accuracy: " + str(accuracy_score(truth, pred)))
# print("f1_score: " + str(f1_score(truth, pred)))
# # print("weighted Roc score: " + str(roc_auc_score(truth,pred,multi_class='ovr',average='weighted')))
# # print("Precision: "+ str(precision_score(truth, pred, average='macro')))
# # print("Recall: "+ str(recall_score(truth, pred, average='macro')))
# # print("Accuracy: " + str(accuracy_score(truth, pred)))
# # print("Macro Roc score: " + str(roc_auc_score(y_true,y_prob,multi_class='ovr',average='macro')))
# from sklearn.metrics import RocCurveDisplay,f1_score
import matplotlib.pyplot as plt
# import numpy as np
from sklearn import metrics

# # print(metrics.accuracy_score(truth,pred))
# # print(metrics.roc_auc_score(truth,pred))
fpr, tpr, thresholds = metrics.roc_curve(truth, pred)


def DrawROC(fpr, tpr, roc_auc, title='ROC curve'):
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % roc_auc, lw=2)
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('%s' % title, fontsize=18)
    plt.legend(loc='lower right')
    plt.show()


# # print(thresholds)
roc_auc = metrics.auc(fpr, tpr)
# DrawROC(fpr,tpr,roc_auc)
# print('auc:',roc_auc)
# # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='1')
# # display.plot()
# # plt.show()

# 绘制混淆矩阵图方法1
import seaborn as sns

confusion = confusion_matrix(truth, pred, labels=[0, 1], normalize=None)  # 计算混淆矩阵
print(confusion)
TN ,FP ,FN ,TP= confusion.ravel()
confusion=[[TP,FN],[FP,TN]]
labels = ['MCN', 'SCN']
plt.figure(figsize=(6, 5))
sns.heatmap(confusion, cmap='YlGnBu', annot=True,
            annot_kws={'fontproperties': 'Times New Roman','size': 20, 'weight': 'bold',
                       }, xticklabels=labels,square=True,
            yticklabels=labels)  # 绘制混淆矩阵
plt.yticks(fontproperties='Times New Roman',size=14,rotation='horizontal')
plt.xticks(fontproperties='Times New Roman',size=14)
title = 'DenseNet201+CBAM'
plt.title(title, fontproperties='Times New Roman', size=18)
plt.xlabel('Predict',fontproperties='Times New Roman', size=18)
plt.ylabel('True',fontproperties='Times New Roman', size=18)
plt.tight_layout()
plt.savefig('./Figure/' + title + '.png', dpi=600)
plt.show()
