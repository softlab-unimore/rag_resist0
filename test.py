import os
import logging
import time
import argparse
import pandas as pd
import numpy as np
import pickle as pkl

from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

from utils import init_args
from main import run

load_dotenv()

logging.basicConfig(filename="./tests/log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.test")

def init_args():
    parser = argparse.ArgumentParser(prog='BPER Table Extractor', description='Extract tables from companies non financial statements')

    parser.add_argument('-m', '--method', choices=["page"], default="page", type=str,
                        help='extraction method')
    parser.add_argument('-p', '--pdf', type=str, required=False, default='',
                        help='relative URI of the pdf file to analyze')
    parser.add_argument('-q', '--query', type=str, required=False, default='',
                        help=' query to be sent to the vector store')
    parser.add_argument('-e', '--embed', action="store_true", default=False, 
                        help='embed documents and load them into the vector store')
    parser.add_argument('-c', '--checkpoint_rate', type=int, required=False, default=1000,
                        help='specify the amount of steps before checkpointing the results')
    parser.add_argument('-l', '--load_from_checkpoint', action='store_true', default=False,
                        help='run the tests starting from the latest checkpoint')

    args = vars(parser.parse_args())

    return args

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

   top_k_labels = [1,2,5,10,20,50]
   if not args["load_from_checkpoint"]:
      acc = [[], [], [], [], [], []]
      result_df = [[]]
      last_iter = 0
      for label in top_k_labels:
         result_df[0].append(f"top@{label} accuracy")
   else:
      max_value = 0
      for file_name in os.listdir("./tests/checkpoint/"):
         file_split = file_name.split(".")
         if file_split[-1] != "csv":
            continue
         value = int(file_split[0])

         if value > max_value:
            max_value = value

      result_df = pd.read_csv(f"./tests/checkpoint/{max_value}.csv").values.tolist()
      with open(f"./tests/checkpoint/{max_value}.pkl", "rb") as reader:
         acc = pkl.load(reader)
      with open(f"./tests/checkpoint/k.pkl", "rb") as reader:
         last_iter = pkl.load(reader)
      logger.info(f"[{datetime.now()}] Starting from checkpoint {max_value}")

   start_time = time.time()
   logger.info(f"[{datetime.now()}] Started testing...")
   for k, (_,row) in enumerate(tqdm(df.iterrows())):
      file_path = f"pdfs/{row['Nome PDF'].iloc[0]}"

      if k < last_iter:
         continue
      if not os.path.exists(file_path):
         #raise FileNotFoundError(f"File {file_path} cannot be found")
         logger.warning(f"[{datetime.now()}] File {file_path} cannot be found")
         continue
      
      args["pdf"] = file_path
      args["query"] = str(row["Descrizione"].iloc[0])

      result = run(args)

      top_1 = [result[0].metadata["page"] + 1]
      top_2 = [result[i].metadata["page"] + 1 for i in range(2)]
      top_5 = [result[i].metadata["page"] + 1 for i in range(5)]
      top_10 = [result[i].metadata["page"] + 1 for i in range(10)]
      top_20 = [result[i].metadata["page"] + 1 for i in range(20)]
      top_50 = [result[i].metadata["page"] + 1 for i in range(50)]

      new_row = []
      for j, res in enumerate([top_1, top_2, top_5, top_10, top_20, top_50]):
         if int(row["Valore"]["Pagina"]) in res:
            acc[j].append(1)
            new_row.append(1)
         else:
            acc[j].append(0)
            new_row.append(0)
      
      result_df.append(new_row)

      if k % args["checkpoint_rate"] == 0:
         with open(f"./tests/checkpoint/{k}.pkl", "wb") as output_file:
            pkl.dump(acc, output_file)
         with open(f"./tests/checkpoint/k.pkl", "wb") as output_file:
            pkl.dump(k, output_file)
         tmp_df = pd.DataFrame(result_df)
         tmp_df.to_csv(f"./tests/checkpoint/{k}.csv", index=False)
         del tmp_df

   end_time = time.time()
   logger.info(f"[{datetime.now()}] Finished testing in {end_time - start_time}")

   for label_k, acc_k in zip(top_k_labels, acc):
      perc = round(sum(acc_k) / len(acc_k), 2)
      logger.info(f"Top@{label_k} accuracy is {perc}")

   #saving results
   new_df = pd.DataFrame(result_df)
   result_df.to_csv(f"./tests/result_{datetime.now()}", index=False)