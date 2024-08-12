import os
import logging
import time
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

from utils import init_args
from main import run

load_dotenv()

logging.basicConfig(filename="./tests/log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.test")

def load_df(path: str) -> pd.DataFrame:
   if path.split(".")[-1] != "csv":
      return pd.DataFrame()
   
   start_time = time.time()
   logger.info(f"[{datetime.now()}] Started loading {path}...")
   df = pd.read_csv(path, header=[0,1])
   columns = pd.DataFrame(df.columns.tolist())

   columns.loc[columns[0].str.startswith('Unnamed:'), 0] = np.nan
   columns.loc[columns[1].str.startswith('Unnamed:'), 1] = np.nan

   columns[0] = columns[0].fillna(method='ffill')
   columns[1] = columns[1].fillna(method='ffill')
   columns[1] = columns[1].fillna('')

   columns = pd.MultiIndex.from_tuples(list(columns.itertuples(index=False, name=None)))
   df.columns = columns
   df = df[df["Valore"]["Origine dato"] == "TABELLA"]

   end_time = time.time()
   logger.info(f"[{datetime.now()}] Loaded {path} in {end_time - start_time}")

   return df


if __name__ == "__main__":
   args = init_args()
   if not os.path.isdir(args["pdf"]):
      raise NotImplementedError(f"The path {args['pdf']} is not a directory. Only directories are supported for testing")

   df_path = "tests/data.csv"
   df = load_df(df_path)

   acc = [[], [], [], [], []]
   top_k_labels = [1,2,5,10,20]
   result_df = [[]]
   for label in top_k_labels:
      result_df[0].append(f"top@{label} accuracy")

   start_time = time.time()
   logger.info(f"[{datetime.now()}] Started testing...")
   for i,row in tqdm(df.iterrows()):
      file_path = f"pdfs/{row['Nome PDF'].iloc[0]}"
      if not os.path.exists(file_path):
         #raise FileNotFoundError(f"File {file_path} cannot be found")
         logger.warning(f"[{datetime.now()}] File {file_path} cannot be found")
      
      args["pdf"] = file_path
      args["query"] = row["Descrizione"].iloc[0]

      result = run(args)

      top_1 = [result[0].metadata["page"] + 1]
      top_2 = [result[i].metadata["page"] + 1 for i in range(2)]
      top_5 = [result[i].metadata["page"] + 1 for i in range(5)]
      top_10 = [result[i].metadata["page"] + 1 for i in range(10)]
      top_20 = [result[i].metadata["page"] + 1 for i in range(20)]
      
      new_row = []
      for i, res in enumerate([top_1, top_2, top_5, top_10, top_20]):
         if int(row["Valore"]["Pagina"]) in res:
            acc[i].append(1)
            new_row.append(1)
         else:
            acc[i].append(0)
            new_row.append(0)
      
      result_df.append(new_row)

   end_time = time.time()
   logger.info(f"[{datetime.now()}] Finished testing in {end_time - start_time}")

   for label_k, acc_k in zip(top_k_labels, acc):
      perc = round(sum(acc_k) / len(acc_k), 2)
      logger.info(f"Top@{label_k} accuracy is {perc}")

   #saving results
   new_df = pd.DataFrame(result_df)
   result_df.to_csv(f"./tests/result_{datetime.now()}", index=False)