# -- coding: utf-8 --
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
def calculate_metric(gt, pred,verbose=False):
    TN ,FP ,FN ,TP= confusion_matrix(gt, pred).ravel()
    # print(TN ,FP ,FN ,TP)
    auc = roc_auc_score(gt, pred)
    acc=(TP + TN) / float(TP + TN + FP + FN)
    sen=TP / float(TP + FN)
    spe=TN / float(TN + FP)
    pre=TP / float(TP + FP)
    f1=f1_score(gt, pred)
    if verbose:
        print('AUC:', auc)
        print('Accuracy:', acc)
        print('Sensitivity:', sen)
        print('Specificity:', spe)
        print('precision',pre)
        print('f1-score:',f1)
    return auc,acc,sen,spe,pre,f1
