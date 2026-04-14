import pandas as pd
from autogluon.tabular import TabularPredictor

from common import load_labeled_data, merge_and_shuffle, split_df


random_state = 2021
mcn_data, scn_data = load_labeled_data('MCN_data_select.csv', 'SCN_data_select.csv', random_state=random_state)
data = merge_and_shuffle(mcn_data, scn_data, random_state=random_state)
data = data.drop(columns='Unnamed: 0')

train_data, test_data = split_df(data, 0.8)
print(train_data.shape)
print(train_data.head())

label = 'label'
print('Summary of class variable: \n', data[label].describe())

save_path = 'agModels-predictClass'
predictor = TabularPredictor(label=label, path=save_path, eval_metric='roc_auc').fit(
    train_data,
    hyperparameters=None,
    num_stack_levels=1,
    num_bag_folds=3,
    time_limit=2000,
)

print(predictor.distill())

y_test = test_data[label]
test_data_nolab = test_data.drop(columns=[label])
print(test_data_nolab.head())

predictor = TabularPredictor.load(save_path)
y_pred = predictor.predict(test_data_nolab)
print('Truth:\n', y_test)
print('Predictions:\n', y_pred)
print(predictor.leaderboard(test_data, silent=True))

results = predictor.fit_summary(show_plot=True)
print(results)
print(predictor.feature_importance(test_data, subsample_size=None))
