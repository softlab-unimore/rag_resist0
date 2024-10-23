import pandas as pd
import numpy as np
import argparse
import os

def init_args():
    parser = argparse.ArgumentParser(prog='BPER Debugger', description='Debug the results of the RAG pipeline')

    parser.add_argument('-c', '--csv', type=str, required=True,
                        help='relative URI of the csv file containing the retrieval results')
    args = vars(parser.parse_args())

    return args

args = init_args()
#csv = "checkpoint/paraphrase-multilingual-MiniLM-L12-v2_tf_idf_0.3_mixed/paraphrase-multilingual-MiniLM-L12-v2_tf_idf_0.3_mixed_420.csv"
#csv = "checkpoint/all-mpnet-base-v2_tf_idf_1.0/all-mpnet-base-v2_tf_idf_1.0_200.csv"
#csv = "paraphrase-multilingual-MiniLM-L12-v2_result.csv"
#csv = "tf_idf_result.csv"
#csv = "multilingual-e5-large-instruct_result.csv"

csv = args["csv"]
df1 = pd.read_csv(csv)

new_header = df1.iloc[0]
df1 = df1[1:]
df1.columns = new_header

print(len(pd.to_numeric(df1["top@20 accuracy"])))

print(f'top@1 accuracy: {round(pd.to_numeric(df1["top@1 accuracy"]).sum() / len(pd.to_numeric(df1["top@1 accuracy"])),2)}')
print(f'top@2 accuracy: {round(pd.to_numeric(df1["top@2 accuracy"]).sum() / len(pd.to_numeric(df1["top@2 accuracy"])),2)}')
print(f'top@5 accuracy: {round(pd.to_numeric(df1["top@5 accuracy"]).sum() / len(pd.to_numeric(df1["top@5 accuracy"])),2)}')
print(f'top@10 accuracy: {round(pd.to_numeric(df1["top@10 accuracy"]).sum() / len(pd.to_numeric(df1["top@10 accuracy"])),2)}')
print(f'top@20 accuracy: {round((pd.to_numeric(df1["top@20 accuracy"]).sum()) / (len(pd.to_numeric(df1["top@20 accuracy"]))),2)}')
print(f'top@50 accuracy: {round(pd.to_numeric(df1["top@50 accuracy"]).sum() / len(pd.to_numeric(df1["top@50 accuracy"])),2)}')
