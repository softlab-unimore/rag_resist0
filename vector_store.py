import torch
import os
import functools
import logging
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from typing import Union

from connectors import PgVectorConnector

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.vector_store")

class VectorStoreHandler:
    def __init__(self):
        #self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

        if torch.cuda.is_available():
            model_kwargs = {"device": "cuda"}
        else:
            model_kwargs = {"device": "cpu"}

        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=model_kwargs)

        self.pgconnector = PgVectorConnector()

    @functools.cache
    def get_vector_store(self, collection_name="coll"):
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=self.pgconnector.get_connection(),
            use_jsonb=True,
        )

        return self.vector_store

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
        logger.info(f"Adding {len(docs)} documents into the vector store...")
        self.vector_store.add_documents(docs)
        end_time = time.time()
        logger.info(f"Added {len(docs)} documents in {end_time - start_time} seconds")

    def delete_from_vector_store(self, ids: Union[list[str], str], collection_name="coll"):
        start_time = time.time()
        logger.info(f"Removing {ids} documents from the vector store...")
        if isinstance(ids, str):
            if ids != "all":
                raise ValueError(f"{ids} is not a valid index. Please provide a list of indices or the \"all\" string")
            self.vector_store.delete_collection(collection_name)
        else:
            self.vector_store.delete(ids=ids)
        end_time = time.time()
        logger.info(f"Removed {ids} documents in {end_time - start_time} seconds")

    @functools.cache
    def query_by_similarity(self, query, k=10, filter={}):
        return self.vector_store.similarity_search(query, k=k, filter=filter)