import pandas as pd

df = pd.read_csv("520.csv")

new_header = df.iloc[0]
df = df[1:]
df.columns = new_header

print(df.head())
print(pd.to_numeric(df["top@20 accuracy"]).sum())
print(len(pd.to_numeric(df["top@50 accuracy"])))
print(round(pd.to_numeric(df["top@20 accuracy"]).sum() / len(pd.to_numeric(df["top@20 accuracy"])),2))