import pandas as pd
import os
from tqdm import tqdm

# closedticket_fname = 'data/small.csv'
closedticket_fname = 'data/closedtickets1.csv'
trans_df = pd.read_csv(closedticket_fname)

def dump_keys(trans_df):
  """ dump keys from the file """
  tkeys = trans_df.keys()
  kfi = open('keys.txt', 'w')
  [kfi.write(tkeys[i]+'\n') for i in range(len(tkeys))] 
  kfi.close()

# uncomment to dump keys to keys.txt file
# dump_keys(trans_df)

def dump_each_column(trans_df, outdir, header):
  """ function to dump all headers values to different files (file names: header.csv)
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
  fname = os.path.join(path, header+'.csv')

  header_list = [header]
  trans_df.to_csv(fname, columns=header_list)

# to dump all the separate columns from the trans_df (closedtickets1.csv)
for header in tqdm(trans_df.keys()):
  dump_each_column(trans_df, 'output_dir', header)


