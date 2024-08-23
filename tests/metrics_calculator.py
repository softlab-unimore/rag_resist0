import pandas as pd
import numpy as np
import os

df1 = pd.read_csv("all-mpnet-base-v2_tf_idf_result_2024-08-23 14:44:40.060981.csv")

new_header = df1.iloc[0]
df1 = df1[1:]
df1.columns = new_header

print(f'top@1 accuracy: {round(pd.to_numeric(df1["top@1 accuracy"]).sum() / len(pd.to_numeric(df1["top@1 accuracy"])),2)}')
print(f'top@2 accuracy: {round(pd.to_numeric(df1["top@2 accuracy"]).sum() / len(pd.to_numeric(df1["top@2 accuracy"])),2)}')
print(f'top@5 accuracy: {round(pd.to_numeric(df1["top@5 accuracy"]).sum() / len(pd.to_numeric(df1["top@5 accuracy"])),2)}')
print(f'top@10 accuracy: {round(pd.to_numeric(df1["top@10 accuracy"]).sum() / len(pd.to_numeric(df1["top@10 accuracy"])),2)}')
print(f'top@20 accuracy: {round(pd.to_numeric(df1["top@20 accuracy"]).sum() / len(pd.to_numeric(df1["top@20 accuracy"])),2)}')
print(f'top@50 accuracy: {round(pd.to_numeric(df1["top@50 accuracy"]).sum() / len(pd.to_numeric(df1["top@50 accuracy"])),2)}')
