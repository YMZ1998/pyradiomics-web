import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
def split_df(df, ratio):
    #用来分割数据集，保留一定比例的数据集当做最终的测试集
    cut_idx = int(round(ratio * df.shape[0]))
    print(cut_idx)
    train_data, test_data = df.iloc[:cut_idx], df.iloc[cut_idx:]
    return (train_data, test_data)
random_state=2021
MCN_data = pd.read_csv('MCN_data_select.csv')
SCN_data = pd.read_csv('SCN_data_select.csv')
MCN_data.insert(0, 'label', 1)  # 插入标签
SCN_data.insert(0, 'label', 0)  # 插入标签
MCN_data = MCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
SCN_data = SCN_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
data = pd.concat([MCN_data, SCN_data])
data=data.drop(columns='Unnamed: 0')
data=data.sample(frac=1.0,random_state=random_state)
train_data, test_data=split_df(data,0.8)
print(train_data.shape)
print(train_data.head())
label = 'label'
print("Summary of class variable: \n", data[label].describe())
save_path = 'agModels-predictClass'  # specifies folder to store trained models
predictor = TabularPredictor(label=label, path=save_path,eval_metric='roc_auc').fit(train_data,hyperparameters=None,num_stack_levels=1,num_bag_folds=3,time_limit=2000)
distilled_model_names = predictor.distill()
print(distilled_model_names)
y_test = test_data[label]  # values to predict
test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
print(test_data_nolab.head())
predictor = TabularPredictor.load(save_path)
y_pred = predictor.predict(test_data_nolab)
print("Truth:\n",y_test)
print("Predictions:\n", y_pred)
# perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=False)
print(predictor.leaderboard(test_data, silent=True))
results = predictor.fit_summary(show_plot=True)
print(results)
print(predictor.feature_importance(test_data,subsample_size=None))