import pandas as pd
import argparse

def init_args():
    parser = argparse.ArgumentParser(prog='BPER Debugger', description='Debug the results of the RAG pipeline')

    parser.add_argument('-c', '--csv1', type=str, required=True,
                        help='relative URI of the csv file containing the retrieval results')
    parser.add_argument('-C', '--csv2', type=str, required=True,
                        help='relative URI of the csv file containing metadata of the results')
    parser.add_argument('-p', '--pdf_error', action="store_true", default=False,
                        help='extract statistics on how many times a pdf has been mislabeled')
    parser.add_argument('-g', '--gri_error', action="store_true", default=False,
                        help='extract statistics on how many times a GRI has been mislabeled')
    parser.add_argument('-d', '--description_error', action="store_true", default=False,
                        help='extract statistics on how many times a description has been mislabeled')
    parser.add_argument('-r', '--riga_error', action="store_true", default=False,
                        help='extract statistics on how many times a row has been mislabeled')
    parser.add_argument('-n', '--normalize', action="store_true", default=False,
                        help='normalize the metrics')
    args = vars(parser.parse_args())

    return args

def get_pdf_error(df1, df2, key="top@50 accuracy", normalize=False):
    mislabeled_df = df1[df1[key] == "0"]
    mislabeled_md = df2.loc[mislabeled_df.index, "Nome PDF"]

    freq = mislabeled_md.value_counts()

    if normalize:
        freq_tot = df2["Nome PDF"].value_counts()
        freq = freq / freq_tot
        freq = freq.fillna(0).sort_values(ascending=False)
    return freq

def get_gri_error(df1, df2, key="top@50 accuracy", normalize=False):
    mislabeled_df = df1[df1[key] == "0"]
    mislabeled_md = df2.loc[mislabeled_df.index, "GRI"]
    freq = mislabeled_md.value_counts()

    if normalize:
        freq_tot = df2["GRI"].value_counts()
        freq = freq / freq_tot
        freq = freq.fillna(0).sort_values(ascending=False)
    return freq

def get_description_error(df1, df2, key="top@50 accuracy", normalize=False):
    mislabeled_df = df1[df1[key] == "0"]
    mislabeled_md = df2.loc[mislabeled_df.index, "Descrizione"]

    freq = mislabeled_md.value_counts()

    if normalize:
        freq_tot = df2["Descrizione"].value_counts()
        freq = freq / freq_tot
        freq = freq.fillna(0).sort_values(ascending=False)

    return freq

def get_riga_error(df1, df2, key="top@50 accuracy", normalize=False):
    mislabeled_df = df1[df1[key] == "0"]
    mislabeled_md = df2.loc[mislabeled_df.index, ["Nome PDF", "Pagina"]]

    freq = mislabeled_md.value_counts()

    return freq

switch_fn = {
    "pdf_error": get_pdf_error,
    "gri_error": get_gri_error,
    "description_error": get_description_error,
    "riga_error": get_riga_error
}

if __name__ == "__main__":
    args = init_args()
    df1 = pd.read_csv(args["csv1"])
    df2 = pd.read_csv(args["csv2"])

    new_header = df1.iloc[0]
    df1 = df1[1:]
    df1.columns = new_header

    new_header = df2.iloc[0]
    df2 = df2[1:]
    df2.columns = new_header

    r = []
    for k,v in args.items():
        if "error" not in k:
            continue
        if isinstance(v, bool) and v:
            result = switch_fn[k](df1, df2, normalize=args["normalize"], key="top@10 accuracy")
            r.append(result.sum())
            print(f"{k}: {result}")
            print()
            print("----------------------------------------")
            print()

    print(r)
