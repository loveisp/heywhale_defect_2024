import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv('./data/train/文件标签汇总数据.csv')
fname = (df.fileName.str.split('.').map(lambda x: x[0]) + '.png').rename('fname')
labels = df.defectType.rename('labels')
df_folds = pd.concat([fname, labels], axis=1)
df_folds['fold'] = 0
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
for i, (train_index, test_index) in enumerate(skf.split(df_folds.labels, df_folds.labels)):
    df_folds.loc[test_index, 'fold'] = i
df_folds.to_csv('./src/split_5_folds.csv', index=None)