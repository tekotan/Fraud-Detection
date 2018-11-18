""" Create ML training data based on rules and augment """
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from de.select_columns import select_columns

class LossPrevTrainingDataPrep(object):
  def __init__(self, redshift_trans_csv_fname, \
    ml_features_columns_fname, \
    da_features_columns_fname, output_dir):
    """ constructor
    Args:
      redshift_trans_csv_fname (str): original redshift dump file name \
        with all transactions
      ml_features_columns_fname (str): all the features required \
        for ML training
      da_features_columns_fname (str): all the features required \
        for data augmentation. Will use these \
        columns to apply rules to generate fraud \
        data for training
      output_path (str): absolute path to dump output
    """
    self.redshift_trans_csv_fname = redshift_trans_csv_fname
    self.ml_features_columns_fname = ml_features_columns_fname
    self.da_features_columns_fname = da_features_columns_fname
    self.output_path = os.path.abspath(output_dir)
    if not os.path.exists(self.output_path):
      os.makedirs(self.output_path)

    # selected columns file name
    self.da_features_select_fname = 'da_select_' + \
      os.path.basename(self.redshift_trans_csv_fname)
    self.ml_features_select_fname = 'ml_select_' + \
      os.path.basename(self.redshift_trans_csv_fname)

  def append_labels_to_final_training_data(self, fraud_transactions):
    """ Function appends label to produce final training data
    """
    # Read 'ml_select_<>.csv' file
    fname = os.path.join(self.output_path, \
      self.ml_features_select_fname)

    select_df = pd.read_csv(fname, \
      skip_blank_lines=True, \
      warn_bad_lines=True, error_bad_lines=False)

    # add a column to store label
    select_df['label'] = 0

    # Append it to ml_select_with_label_<>.csv file and dump
    for ticketid in tqdm(fraud_transactions):
      ticketid = int(ticketid[0])
      select_df.loc[select_df['ticketnum'] == ticketid, 'label']=1
  
    # dump the file
    fname = os.path.join(self.output_path, \
      'ml_select_with_label_' + \
      os.path.basename(self.redshift_trans_csv_fname))

    select_df.to_csv(fname)

  def generate_fraud_training_data(self):
    """ Function generaes fraud data based on rules
    Args:

    Returns:

    Description:
      What is a fraud:
      Same credit card (paycode, ccdaccount, ccdexupdate) \
        used at the same facility (locat) in the \
        same month (dtout) for a different vehicle \
        (make, color, plate)

      Fetch below columns from table:
      locat, dtout (format?), paycode (3,4,6,10), make , \
      color , plate, ccdaccount , ccdexpdate , \
      ratedescription (only pick transactions \
      which are not 'default')

      [ Above fileds are already extracted to 'da_select_<>.csv' ]
    """
    # Read 'da_select_<>.csv' file
    fname = os.path.join(self.output_path, \
      self.da_features_select_fname)

    select_df = pd.read_csv(fname, \
      skip_blank_lines=True, \
      warn_bad_lines=True, error_bad_lines=False)

    # add a column to store label
    select_df['label'] = 0

    # Ignore default transaction from ratedescription column
    select_df = select_df[select_df['ratedescription'] != 'Default']
    select_df = select_df[select_df['ratedescription'] != 'default']

    # Ignore transactions which are not credit card based, and only 
    # have paycode of 3, 4, 6, 10
    paycode = select_df['paycode']
    paycode_logical_or = np.logical_or(np.logical_or(np.logical_or( \
      paycode == 3, paycode == 4), \
      paycode == 6), paycode == 10)
    select_df = select_df[paycode_logical_or]
    # select_df.to_csv(os.path.join(self.output_path, 'da_select_debug.csv'))

    # Ignore nan 
    select_df = select_df.dropna(subset=['ccdaccount'])

    ## Apply rules to generate label

    location_dict = {}
    def build_fraud_transaction_map(row):
      """ Same credit card (paycode, ccdaccount, ccdexpdate) \
        used at the same facility (locat) in the \
        same month (dtout) for a different vehicle \
        (make, color, plate)
      """
      ticketid_tup = (row['ticketnum'])
      car_tup = (ticketid_tup, \
        (row['make'], row['color'], row['plate']))
      cc_tup = (row['ccdaccount'], row['ccdexpdate'])
      loc_tup = (row['locat']) # can include other stuff
      if loc_tup not in location_dict.keys():
        location_dict[loc_tup]={}
      if cc_tup not in location_dict[loc_tup].keys():
        location_dict[loc_tup][cc_tup] = []
      location_dict[loc_tup][cc_tup].append(car_tup)

    select_df.apply(build_fraud_transaction_map, axis=1)

    fraud_transactions=[]
    ## Apply rules to generate label
    for loc in location_dict.keys():
      for cc in location_dict[loc].keys():
        car_list = location_dict[loc][cc]
        if len(car_list) > 1:
          # fraud transaction
          for car in car_list:
            fraud_transactions.append(car)

    # Append it to da_select_with_label_<>.csv file and dump
    for ticketid in tqdm(fraud_transactions):
      ticketid = int(ticketid[0])
      select_df.loc[select_df['ticketnum'] == ticketid, 'label']=1
    
    # dump the file
    fname = os.path.join(self.output_path, \
      'da_select_with_label_' + \
      os.path.basename(self.redshift_trans_csv_fname))
    select_df.to_csv(fname)

    return fraud_transactions

  def training_data_prep(self):
    """ function to create fradulant training dataset.
    Args:
      None
    Return:
      None

    Description:
      Step-0: Read original redshift dump csv file with all \
        transactions
      Step-1: Dump file 'da_select_<>.csv' to have all \
        feature columns for data augmentation
      Step-2: Apply rules to identify if a transaction is fraud \
        and create labels and dump another file \
        'da_select_with_label_<>.csv'
      Step-3: Dump file 'ml_select_<>.csv' to have all \
        feature columns required for ml
      Step-4: Append labels from Step-3 to ml_select_with_label_<>.csv
    """
    # Step-1: 
    # select columns based on self.da_features_columns_fname
    select_columns(self.redshift_trans_csv_fname, \
      self.da_features_columns_fname, \
      self.output_path, \
      output_fname=self.da_features_select_fname)

    # Step-2:
    # Apply rules to get fraud data
    fraud_transactions = self.generate_fraud_training_data()

    # Step-3:
    # select columns based on self.ml_features_columns_fname
    select_columns(self.redshift_trans_csv_fname, \
      self.ml_features_columns_fname, \
      self.output_path, \
      output_fname=self.ml_features_select_fname)

    # Step-4:
    # Augment labels to final trainig data
    self.append_labels_to_final_training_data(fraud_transactions)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--redshift_trans_csv_fname', \
    type=str, required=True, \
    help="Original transaction csv file from redshift dump")
  parser.add_argument('-ml_f', '--ml_features_columns_fname', type=str, \
    required=True, help="Text file to have ML features' headers")
  parser.add_argument('-da_f', '--data_aug_features_columns_fname', \
    type=str, required=True, \
    help="Text file to have ML features' headers")
  parser.add_argument('-o', '--output_dir', type=str, \
    required=True, help="Output dir")
  args = parser.parse_args()

  # uncomment to dump keys to keys.txt file
  # dump_keys(trans_df)

  redshift_trans_csv_fname = args.redshift_trans_csv_fname
  ml_features_columns_fname = args.ml_features_columns_fname
  da_features_columns_fname = args.data_aug_features_columns_fname

  # Prepare output directory
  output_dir = args.output_dir

  trn_dataprep = LossPrevTrainingDataPrep(redshift_trans_csv_fname, \
    ml_features_columns_fname, \
    da_features_columns_fname, \
    output_dir)
  trn_dataprep.training_data_prep()


