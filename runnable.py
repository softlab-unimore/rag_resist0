import logging
import time
import traceback

from dataprocessor import PageProcessor
from vector_store import VectorStoreHandler, SparseStoreHandler, EnsembleRetrieverHandler
from llm import LlamaModel, OpenAIModel
from table_extraction import UnstructuredTableExtractor, TabulaTableExtractor, CombinedTableExtractor, Docdetection

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.main")

class Runnable:

    def __init__(self, args):
        if not args["fast"]:
            self.table_extractor = Docdetection()

        if args["use_dense"]:
            self.vsh = VectorStoreHandler(args)
        elif args["use_sparse"]:
            self.ssh = SparseStoreHandler(args)
        elif args["use_ensemble"]:
            self.ens = EnsembleRetrieverHandler(args)
        if args["use_llama"]:
            self.extr_model = LlamaModel(args)
        elif args["use_openai"]:
            self.extr_model = OpenAIModel()

        self.processor = PageProcessor()

        self.args = args

    def run_value_extraction(self, contents):
        contents_txt = [doc.page_content for doc in contents]
        if not self.args["fast"]:
            tables_html = self.table_extractor.extract_table(contents)
        else:
            tables_html = [[] for _ in contents_txt]

        try:
            results = self.extr_model.run(contents_txt, self.args["query"], tables_html)
        except:
            logger.warning(f"OpenAI has returned an error")
        results = {(content.metadata["source"], str(content.metadata["page"])): eval(result) for content, result in zip(contents, results)}
        return results

    def run(self):
        if self.args["use_dense"]:
            self.vsh.get_vector_store()

        contents = self.processor.get_pdf_content(self.args["pdf"])

        if self.args["embed"]:
            if self.args["use_dense"]:
                self.vsh.load_docs_in_vector_store(contents)
            elif self.args["use_sparse"]:
                self.ssh.load_docs_in_sparse_store(contents)
            similar_docs = None
        elif self.args["query"]:
            self.args["k"] = int(self.args["k"])
            filters = (("source", self.args["pdf"]), ("model_name", self.args["model_name"]))
            if self.args["use_dense"]:
                similar_docs = self.vsh.query_by_similarity(self.args["query"], k=self.args["k"], filters=filters, with_scores=True)
            elif self.args["use_sparse"]:
                similar_docs = self.ssh.query_by_similarity(self.args["query"], k=self.args["k"], source=self.args["pdf"], with_scores=True)
            elif self.args["use_ensemble"]:
                similar_docs = self.ens.query_by_similarity(self.args["query"], k=self.args["k"], filters=filters)

        return similar_docs
