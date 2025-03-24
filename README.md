# Indicator Extractor



This repository contains the code to extract **financial and non-financial indicators** from company reports (pdf files)  



## How to run



### Setup



Go to the root of the directory, create a conda virtual environment and activate it

```
conda create -n env
conda activate env
```

Install the [Tesseract OCR Engine](https://github.com/tesseract-ocr/tesseract), needed to use one of [Unstructured](https://github.com/Unstructured-IO/unstructured) or [Deepdoctection](https://github.com/deepdoctection/deepdoctection) for table extraction.  
Then, copy the file `sample_config/.env` to the root of the repository and fill the missing values.  
Build and run the docker-compose file. The container hosts the pgvector database containing the embeddings extracted from the pdf files

```
sudo docker compose build
sudo docker compose up
```



### Useful python commands

#### Create dense embeddings

To store the semantic embeddings in the database, run

```
PYTHONHASHSEED=0 python3 main.py --pdf [PDF_PATH] --embed --use_dense --model_name [MODEL_NAME]
```

where

1. `PYTHONHASHSEED=0` is an environment variable making the `hash` function **deterministic**. `hash` is used to parse document chunks, producing an **unique id** which is used as primary key inside the embedding database. In this way, the system **avoids the computation of the document embeddings** if the embedding is already stored inside the database;
2. `PDF_PATH` can be either a single pdf file or a directory storing pdf files;
3. `--embed` and `--use_dense` indicate that the system should embed the documents using the model `MODEL_NAME` (taken from Huggingface). By default, `MODEL_NAME="intfloat/multilingual-e5-large-instruct"`.

#### Create sparse embeddings

To store the data for the sparse embedding, run

```
PYTHONHASHSEED=0 python3 main.py --pdf [PDF_PATH] --embed --use_sparse --syn_model_name [SYN_MODEL_NAME]
```

where

1. `PYTHONHASHSEED=0` same as above;
2. `PDF_PATH` can be either a single pdf file or a directory storing pdf files;
3. `--embed` and `--use_sparse` indicate that the system should embed the documents using the model `SYN_MODEL_NAME`. By default, `SYN_MODEL_NAME="tf_idf"`.

#### Query the embeddings

To query the embeddings, run

```
python3 main.py --pdf [PDF_PATH] --query [QUERY_STRING] --use_dense --model_name [MODEL_NAME] --k [TOP_K_RESULTS]
```

This command will return the top-k document chunks (i.e. document pages) obtained from the dense (semantic) query.

To do the same for the sparse (syntactic) embeddings, run

```
python3 main.py --pdf [PDF_PATH] --query [QUERY_STRING] --use_sparse --syn_model_name [MODEL_NAME] --k [TOP_K_RESULTS]
```

#### Query the embeddings with ensemble method

The ensemble method leverages both semantic and syntactic retrieval modes to further improve the system. To use the ensemble, run

```
python3 main.py --pdf [PDF_PATH] --query [QUERY_STRING] --use_ensemble --model_name [MODEL_NAME] --syn_model_name [SYN_MODEL_NAME] --k [TOP_K_RESULTS] --lambda [LAMBDA_VALUE]
```

The additional parameter `--lambda` is a scalar value that controls the importance of syntactic features over semantic ones. The higher the value, the more we give importance to the `SYN_MODEL_NAME` (e.g. tf_idf)


#### How to be faster by skipping the table extraction

Inside the final prompt that computes the answer, the model extracts by default the tables inside the document pages. This is done to improve the accuracy of the model, but it is slower.
To skip the table extraction phase, use the flag `--skip_table_extraction`.

#### To reproduce the results

Run the file `test.py` with

```
python3 test.py --pdf [PDF_PATH] --use_[dense|sparse|ensemble] --model_name [MODEL_NAME] --syn_model_name [SYN_MODEL_NAME] --checkpoint_rate [CHECKPOINT_RATE]
```

with `--checkpoint_rate` is the saving frequency. The files will be stored in the `tests/` directory
