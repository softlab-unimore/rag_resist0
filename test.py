import os
import logging
import time
import argparse
import re
import pandas as pd
import numpy as np
import pickle as pkl
import ast

from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

from utils import init_args
from runnable import Runnable

load_dotenv()

logging.basicConfig(filename="./tests/log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.test")

def check_args(args):
    if sum([args["use_dense"], args["use_sparse"], args["use_ensemble"]]) != 1:
        raise ValueError(f"only one argument between \"use_dense\", \"use_sparse\" or \"use_ensemble\" must be specified")
    if args["query"] != '' and args["embed"]:
        raise ValueError(f"you can't embed and query at the same time. Please specify only one argument between \"--embed\" and \"--query\"")
    if args["use_ensemble"] and args["embed"]:
        raise ValueError(f"cannot embed using the ensemble model. The ensemble model can only be used for querying, after the vector and sparse stores are loaded into the db")

def init_args():
    parser = argparse.ArgumentParser(prog='BPER Table Extractor', description='Extract tables from companies non financial statements')

    parser.add_argument('-m', '--method', choices=["page"], default="page", type=str,
                        help='extraction method')
    parser.add_argument('-p', '--pdf', type=str, required=True, default='',
                        help='relative URI of the pdf file to analyze')
    parser.add_argument('-q', '--query', type=str, required=False, default='',
                        help=' query to be sent to the vector store')
    parser.add_argument('-e', '--embed', action="store_true", default=False, 
                        help='embed documents')
    parser.add_argument('-d', '--use_dense', action="store_true", default=False, 
                        help='use vector store')
    parser.add_argument('-s', '--use_sparse', action="store_true", default=False, 
                        help='use sparse store')
    parser.add_argument('-E', '--use_ensemble', action="store_true", default=False, 
                        help='use ensemble method')
    parser.add_argument('-M', '--model_name', type=str, required=False, default="sentence-transformers/all-mpnet-base-v2", 
                        help='name of the neural model to use')
    parser.add_argument('-S', '--syn_model_name', type=str, required=False, default="tf_idf", 
                        help='name of the sparse model to use')
    parser.add_argument('-c', '--checkpoint_rate', type=int, required=False, default=1000,
                        help='specify the amount of steps before checkpointing the results')
    parser.add_argument('-l', '--load_from_checkpoint', action='store_true', default=False,
                        help='run the tests starting from the latest checkpoint')
    parser.add_argument('-L', '--lambda', type=float, required=False, default=0.3, 
                        help='integration scalar for syntactic features. Only usable with --use_ensemble')
    parser.add_argument('-u', '--use_llama', action="store_true", required=False, default=False,
                        help='Use llama3.1 8b')
    parser.add_argument('-o', '--use_openai', action="store_true", required=False, default=False,
                        help='Use gpt-4o-mini')

    args = vars(parser.parse_args())
    check_args(args)

    return args

def remove_trailing_zeros(sent):
   if '.' in sent or ',' in sent:
      return sent.rstrip('0').rstrip('.').rstrip(',')
   return sent

def remove_not_numbers(sent):
    return ''.join(char for char in sent if char.isdigit())

def remove_non_numbers_at_end(sent):
    i = len(sent) - 1
    while i >= 0 and not sent[i].isdigit():
        i -= 1
    return sent[:i+1]

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

   if args["use_dense"]:
      abbrv_model_name = args["model_name"].split("/")[-1]
   else:
      abbrv_model_name = args["syn_model_name"].split("/")[-1]

   if args["use_ensemble"]:
      abbrv_model_name = args["model_name"].split("/")[-1]+"_"+args["syn_model_name"].split("/")[-1] + "_"+str(args["lambda"])

   dir_path = f"./tests/checkpoint/{abbrv_model_name}"
   if not os.path.isdir(dir_path):
      os.makedirs(dir_path)

   if not os.path.isdir(args["pdf"]):
      raise NotImplementedError(f"The path {args['pdf']} is not a directory. Only directories are supported for testing")

   df_path = "tests/data.csv"
   df = load_df(df_path)

   top_k_labels = [1,2,5,10,20,50]
   if not args["load_from_checkpoint"]:
      acc = [[], [], [], [], [], []]
      result_df = [[]]
      result_metadata = [["Nome PDF", "GRI", "Descrizione", "Pagina", "Valore"]]
      last_iter = 0
      for label in top_k_labels:
         result_df[0].append(f"top@{label} accuracy")
   else:
      max_value = 0
      for file_name in os.listdir(dir_path):
         file_split = file_name.split(".")
         if file_split[-1] != "csv":
            continue

         file_split = file_split[0].split("_")
         value = int(file_split[1])

         if value > max_value:
            max_value = value

      result_df = pd.read_csv(f"{dir_path}/{abbrv_model_name}_{max_value}.csv").values.tolist()
      result_metadata = pd.read_csv(f"{dir_path}/md_{abbrv_model_name}_{max_value}.csv").values.tolist()
      with open(f"{dir_path}/{abbrv_model_name}_{max_value}.pkl", "rb") as reader:
         acc = pkl.load(reader)
      with open(f"{dir_path}/k_{abbrv_model_name}.pkl", "rb") as reader:
         last_iter = pkl.load(reader)
      logger.info(f"[{datetime.now()}] Starting from checkpoint {max_value}")

   start_time = time.time()
   logger.info(f"[{datetime.now()}] Started testing...")
   correct_pred = []
   top_k = []
   r = Runnable(args)

   #count = 0
   result_pd = []
   for k, (_,row) in enumerate(tqdm(df.iterrows())):
      if k == 420:
         break
      file_path = f"pdfs/{row['Nome PDF'].iloc[0]}"
      logger.info(file_path)

      if not os.path.exists(file_path):
         print(f"[{datetime.now()}] File {file_path} cannot be found")
         logger.warning(f"[{datetime.now()}] File {file_path} cannot be found")
         continue
      """if count < 122:
         count += 1
         continue"""

      args["pdf"] = file_path
      if str(row["Descrizione"].iloc[0]) == "nan":
         args["query"] = str(row["INDICATORE"].iloc[0])
      else:
         args["query"] = str(row["Descrizione"].iloc[0])

      r.args = args

      result = r.run() #r.run(gri_code=str(row["GRI"].iloc[0]))
      if not args["use_ensemble"]:
         result = [r[0] for r in result]
        
      top_1 = [result[0].metadata["page"] + 1]
      top_2 = [result[i].metadata["page"] + 1 for i in range(2)]
      top_5 = [result[i].metadata["page"] + 1 for i in range(5)]
      top_10 = [result[i].metadata["page"] + 1 for i in range(10)]
      top_20 = [result[i].metadata["page"] + 1 for i in range(20)]
      top_50 = [result[i].metadata["page"] + 1 for i in range(50)]

      new_row = []

      for j, res in enumerate([top_1, top_2, top_5, top_10, top_20, top_50]):
         if int(row["Valore"]["Pagina"]) in res:
            if j == 4:
               logger.info(f"Valore trovato: {row['Valore']['Valore testuale']}")
            acc[j].append(1)
            new_row.append(1)
         else:
            if j==4:
               logger.info("Valore non trovato")
            acc[j].append(0)
            new_row.append(0)
         
      result_df.append(new_row)
      result_metadata.append([
         row["Nome PDF"].iloc[0], row["GRI"].iloc[0],
         row["Descrizione"].iloc[0],
         row["Valore"]["Pagina"],
         row["Valore"]["Valore testuale"]
      ])

      if not (args["use_openai"] or args["use_llama"]):
         if k % args["checkpoint_rate"] == 0:
            with open(f"{dir_path}/{abbrv_model_name}_{k}.pkl", "wb") as output_file:
               pkl.dump(acc, output_file)
            with open(f"{dir_path}/k_{abbrv_model_name}.pkl", "wb") as output_file:
               pkl.dump(k, output_file)
            tmp_df = pd.DataFrame(result_df)
            tmp_df.to_csv(f"{dir_path}/{abbrv_model_name}_{k}.csv", index=False)
            tmp_df = pd.DataFrame(result_metadata)
            tmp_df.to_csv(f"{dir_path}/md_{abbrv_model_name}_{k}.csv", index=False)
            del tmp_df

            end_time = time.time()
            logger.info(f"[{datetime.now()}] Finished testing in {end_time - start_time}")

            for label_k, acc_k in zip(top_k_labels, acc):
               perc = round(sum(acc_k) / len(acc_k), 2)
               logger.info(f"Top@{label_k} accuracy is {perc}")

            #saving results
            new_df = pd.DataFrame(result_df)
            print(f"./tests/{abbrv_model_name}_result.csv")
            new_df.to_csv(f"./tests/{abbrv_model_name}_result.csv", index=False)
            new_df = pd.DataFrame(result_metadata)
            new_df.to_csv(f"./tests/md_{abbrv_model_name}_result.csv", index=False)
      else:
         result_llm = r.run_value_extraction([doc for doc in result[:20]])

         found = False
         result_pages = {}
         gold = row['Valore']['Valore testuale']
         #if not isinstance(gold, str):
         gold = remove_not_numbers(remove_trailing_zeros(remove_non_numbers_at_end(str(gold))))
         logger.info(gold)
         for res in result_llm:
            #gold = gold.replace(",",".").strip()
            #try:
            #   gold = float(gold)
            #except:
            #   print(" cannot be casted to float")
            #   continue
               
            try:
               res = ast.literal_eval(res)
               for i in range(len(res)):
                  res[i] = remove_not_numbers(remove_trailing_zeros(remove_non_numbers_at_end(str(res[i]))))

               logger.info(res)
            except:
               logger.warning(f"The result given by OpenAI is not a valid Python value: {res}")
               continue

            try:
               if gold in res:
                  found = True
               if row["Valore"]["Pagina"] not in result_pages.keys():
                  result_pages[row["Valore"]["Pagina"]] = []
               result_pages[row["Valore"]["Pagina"]].extend(res)
            except:
               logger.warning(f"The result given by OpenAI is not an iterable: {res}")

         correct_pred.append(int(found))
         result_pd.append([row["Nome PDF"], row["Valore"]["Pagina"], gold, result_pages, found])

      if k%10 == 0:
         with open("test_results.pkl", "wb") as writer:
            pkl.dump(result_pd, writer)
         print(correct_pred)
         logger.info(f"Partial results: {correct_pred}")
         print(acc[4])
         print()

   result_pd = pd.DataFrame(result_pd)
   result_pd.to_csv("final_test_results.csv", index=False)
   print(correct_pred)
   print(acc)
   print()
   if args["query"]:
      casual_value_extraction = 0
      print(f"Value extraction accuracy: {sum(correct_pred) / len(correct_pred)}")
      print(f"Page filtering accuracy: {sum(acc[4]) / len(acc[4])}")
      print(len(acc[4]))
      for cp, a in zip(correct_pred, acc[4]):
         if cp and not a:
            casual_value_extraction += 1
      print(f"Percentage of value extracted correctly but from wrong pages: {casual_value_extraction / len(correct_pred)}")
      print(len(correct_pred) == len(acc[4]))
      #print(f"Mean top-k for correct predictions: {sum(top_k) / sum(correct_pred)}")
