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

def select_columns(trans_fname, features_fname, output_dir):
    """ Function selects columns
    """
    trans_df = pd.read_csv(trans_fname)
    output_fname = 'select_' + os.path.basename(trans_fname)

    # to dump all the separate columns from the trans_df (closedtickets1.csv)
    # for header in tqdm(trans_df.keys()):
    #  dump_each_column(trans_df, 'output_dir', header)
    dump_each_column(
        trans_df, output_dir, [line.rstrip("\n") for line in open(features_fname)], output_fname
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_csv', type=str, \
            required=True, help="Input csv file")
    parser.add_argument('-f', '--feature_columns', type=str, \
            required=True, help="Text file to have features (headers)")
    parser.add_argument('-o', '--output_dir', type=str, \
            required=True, help="Output dir")
    args = parser.parse_args()

    # uncomment to dump keys to keys.txt file
    # dump_keys(trans_df)

    trans_fname = args.input_csv
    features_fname = args.feature_columns
    output_dir = args.output_dir
    select_columns(trans_fname, features_fname, output_dir)
