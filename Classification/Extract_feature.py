# -- coding: utf-8 --
import pandas as pd
from radiomics import featureextractor
import os

kinds = ['MCN', 'SCN']
para_path = './CT-extractor.yaml'  # 这个是特征处理配置文件，具体可以参考pyradiomics官网网站
extractor = featureextractor.RadiomicsFeatureExtractor(para_path)
for kind in kinds:
    print("{}:开始提取特征".format(kind))
    features_dict = dict()
    df = pd.DataFrame()
    path = './MyData/' + kind
    paths = os.listdir(path)
    # print(paths)
    if '.DS_Store' in paths:  # Mac隐藏文件
        paths.remove('.DS_Store')
    # 使用配置文件初始化特征抽取器
    from tqdm import tqdm
    for folder in tqdm(paths):
        # print(folder)
        for f in os.listdir(os.path.join(path, folder)):
            if 'seg' in f:
                lab_path = os.path.join(path, folder, f)
            else:
                ori_path = os.path.join(path, folder, f)
        features = extractor.execute(ori_path, lab_path)  # 抽取特征
        for key, value in features.items():  # 输出特征
            if 'diagnostics_Versions' in key or 'diagnostics_Configuration' in key:  # 这些都是一些共有的特征，可以去掉
                continue
            features_dict[key] = value
        df = df.append(pd.DataFrame.from_dict(features_dict.values()).T, ignore_index=True)
    df.columns = features_dict.keys()
    # if os.path.exists('{}.csv'):
    #     os.remove('{}.csv')
    df.to_csv('{}.csv'.format(kind), index=0)
print("完成")
