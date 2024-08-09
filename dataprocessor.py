import functools
from PyPDF2 import PdfWriter, PdfReader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader

import logging
import time

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.dataprocessor")

class PageProcessor:
    """
    This class extracts textual contents from pdf files.
    It is used when trying to do RAG on simple pages, without any specific image/table handling
    """

    def __init__(self):
        pass

    @functools.cache
    def _get_reader(self, pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            data = loader.load_and_split()

            return data
        except:
            raise ValueError(f"{pdf_path} is non existent")


    def get_pdf_content(self, pdf_path):
        data = self._get_reader(pdf_path)
        return data
    