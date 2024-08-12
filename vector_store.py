import torch
import os
import functools
import logging
import time
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from typing import Union
from tqdm import tqdm

from connectors import PgVectorConnector

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.vector_store")

class VectorStoreHandler:
    def __init__(self):
        #self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.embeddings = self.get_embeddings(self.model_name, device)

        self.pgconnector = PgVectorConnector()

    @functools.cache
    def get_embeddings(self, model_name, device="cpu"):
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

    @functools.cache
    def get_vector_store(self, collection_name="coll"):
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=self.pgconnector.get_connection(),
            use_jsonb=True,
        )

        return self.vector_store

    def hash_doc(self, doc):
        content = doc.page_content
        metadata = doc.metadata

        input_hash = content + ''.join(list(map(lambda x: str(x), list(metadata.values()))))
        return str(hash(input_hash))

    def load_docs_in_vector_store(self, docs):
        """
        docs: list[str] -> [
            {
                "page_content": "...",
                "metadata": {...}
            }
        ]
        """
        start_time = time.time()
        logger.info(f"[{datetime.now()}] Adding {len(docs)} documents into the vector store...")
        hashes = []
        for doc in docs:
            hashed_input = self.hash_doc(doc)
            hashes.append(hashed_input)
        
        conn = self.pgconnector.start_db_connection()
        existing_ids = self.pgconnector.get_existing_ids(conn, hashes)
        self.pgconnector.close_db_connection(conn)
        
        allowed_docs = []
        allowed_hashes = []
        unallowed_docs = [] # for debugging

        for hash, doc in zip(hashes, docs):
            if hash in existing_ids:
                unallowed_docs.append((doc.metadata["source"], doc.metadata["page"]))
            else:
                allowed_docs.append(doc)
                allowed_hashes.append(hash)
        
        if len(unallowed_docs) > 0:
            logger.info(f"[{datetime.now()}] The following documents {unallowed_docs} are already in the db. Skipping...")

        for hash, doc in tqdm(zip(allowed_hashes, allowed_docs)): #processing docs and hashes one-by-one to prevent db connection drops (https://docs.sqlalchemy.org/en/20/errors.html#error-e3q8)
            self.vector_store.add_documents([doc], ids=[hash])
        end_time = time.time()
        logger.info(f"[{datetime.now()}] Added {len(allowed_docs)} documents in {end_time - start_time} seconds")

    def delete_from_vector_store(self, ids: Union[list[str], str], collection_name="coll"):
        start_time = time.time()
        logger.info(f"[{datetime.now()}] Removing {ids} documents from the vector store...")
        if isinstance(ids, str):
            if ids != "all":
                raise ValueError(f"{ids} is not a valid index. Please provide a list of indices or the \"all\" string")
            self.vector_store.delete_collection(collection_name)
        else:
            self.vector_store.delete(ids=ids)
        end_time = time.time()
        logger.info(f"[{datetime.now()}] Removed {ids} documents in {end_time - start_time} seconds")

    @functools.cache
    def query_by_similarity(self, query, k=20, filters=()):
        d_filter = {}
        if len(filters) > 0:
            d_filter[filters[0]] = filters[1]
        return self.vector_store.similarity_search(query, k=k, filter=d_filter)