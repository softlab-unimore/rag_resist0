import logging
import time

from utils import init_args

from dataprocessor import PageProcessor
from vector_store import VectorStoreHandler

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.main")

switch_method = {
    "page": PageProcessor,
}

if __name__ == "__main__":
    args = init_args()
    try:
        processor = switch_method[args["method"]]()
    except:
        raise ValueError(f"{args['method']} is not a valid extraction method")
    
    vsh = VectorStoreHandler()
    vs = vsh.get_vector_store()
    contents = processor.get_pdf_content(args["pdf"])
    if args["embed"]:
        vsh.load_docs_in_vector_store(contents)
    else:
        similar_docs = vsh.query_by_similarity(args["query"])
        print(similar_docs)
