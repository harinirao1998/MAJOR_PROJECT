import pandas as pd


df_train = pd.read_csv('../../data/raw/labels/train_split_Depression_AVEC2017.csv')

df_test = pd.read_csv('../../data/raw/labels/dev_split_Depression_AVEC2017.csv')

df_dev = pd.concat([df_train, df_test], axis=0)
