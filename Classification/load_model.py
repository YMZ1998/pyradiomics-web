import joblib
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
print(data.columns)
data=data.drop(columns='Unnamed: 0')
data=data.sample(frac=1.0,random_state=random_state)
train_data, test_data=split_df(data,0.8)
y_test = test_data['label']  # values to predict
test_data= test_data.drop(columns=['label'])
save_path = 'agModels-predictClass'
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor.load(save_path)
y_pred = predictor.predict(test_data)
print("Truth:\n",y_test)
print("Predictions:\n", y_pred)
results = predictor.fit_summary(show_plot=True)
