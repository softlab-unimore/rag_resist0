import logging
import time
import traceback

from dataprocessor import PageProcessor
from vector_store import VectorStoreHandler, SparseStoreHandler, EnsembleRetrieverHandler
from llama import LlamaModel, OpenAIModel

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.main")

class Runnable:

    def __init__(self, args):
        self.switch_method = {
            "page": PageProcessor,
        }
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

        try:
            self.processor = self.switch_method[args["method"]]()
        except:
            raise ValueError(f"{args['method']} is not a valid extraction method")

        self.args = args

    def run_value_extraction(self, contents):
        results = self.extr_model.run(contents, self.args["query"])
        return results

    def run(self, gri_code=None):
        if gri_code is not None:
            if gri_code != "nan":
                self.args["use_dense"] = True
                self.args["use_sparse"] = False
            else:
                self.args["use_sparse"] = True
                self.args["use_dense"] = False

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
            filters = (("source", self.args["pdf"]), ("model_name", self.args["model_name"]))
            if self.args["use_dense"]:
                similar_docs = self.vsh.query_by_similarity(self.args["query"], filters=filters, with_scores=True)
            elif self.args["use_sparse"]:
                similar_docs = self.ssh.query_by_similarity(self.args["query"], source=self.args["pdf"], with_scores=True)
            elif self.args["use_ensemble"]:
                similar_docs = self.ens.query_by_similarity(self.args["query"], filters=filters)

        return similar_docs
