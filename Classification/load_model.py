import pandas as pd
from autogluon.tabular import TabularPredictor

from common import load_labeled_data, merge_and_shuffle, split_df


random_state = 2021
mcn_data, scn_data = load_labeled_data('MCN_data_select.csv', 'SCN_data_select.csv', random_state=random_state)
data = merge_and_shuffle(mcn_data, scn_data, random_state=random_state)
print(data.columns)
data = data.drop(columns='Unnamed: 0')

train_data, test_data = split_df(data, 0.8)
y_test = test_data['label']
test_data = test_data.drop(columns=['label'])

save_path = 'agModels-predictClass'
predictor = TabularPredictor.load(save_path)
y_pred = predictor.predict(test_data)
print('Truth:\n', y_test)
print('Predictions:\n', y_pred)

results = predictor.fit_summary(show_plot=True)
