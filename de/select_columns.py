""" file to read original transactions csv file \
        and dump relevant columns """
import pandas as pd
import os
from tqdm import tqdm
import argparse

def dump_keys(trans_df, keys_fname):
    """ dump all keys from the input csv file """
    tkeys = trans_df.keys()
    kfi = open(keys_fname, "w")
    [kfi.write(tkeys[i] + "\n") for i in range(len(tkeys))]
    kfi.close()

def dump_each_column(trans_df, outdir, header, output_fname='dataset.csv'):
    """ function to dump columns specified in feature_headers
  Args:
    trans_df (panda dataframe):
    outdir (str): path where to put the output file
    headers (list): list of headers
  Return:
    None
  """
    path = os.path.abspath(outdir)
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, output_fname)
    header_list = header
    trans_df.to_csv(fname, columns=header_list)

def select_columns(trans_csv_flist, features_fname, output_dir, output_fname):
    """ Function selects columns
    """
    df_list = []
    for csvf in tqdm(trans_csv_flist):
      df = pd.read_csv(csvf, \
          skip_blank_lines=True, \
          warn_bad_lines=False, error_bad_lines=False)
      df_list.append(df)

    trans_df = pd.concat(df_list)


    # to dump all the separate columns from the trans_df
    dump_each_column(
        trans_df, output_dir, [line.rstrip("\n") for line in open(features_fname)], output_fname
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_csv_flist', type=list, \
            required=True, help="Input csv file")
    parser.add_argument('-f', '--feature_columns', type=str, \
            required=True, help="Text file to have features (headers)")
    parser.add_argument('-o', '--output_dir', type=str, \
            required=True, help="Output dir")
    args = parser.parse_args()

    # uncomment to dump keys to keys.txt file
    # dump_keys(trans_df)

    trans_csv_flist = args.input_csv_flist
    features_fname = args.feature_columns
    output_dir = args.output_dir
    select_columns(trans_csv_flist, features_fname, output_dir)
