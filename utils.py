import argparse
import logging

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

    args = vars(parser.parse_args())

    return args