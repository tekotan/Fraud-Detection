""" Create ML training data based on rules and augment """
import argparse
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from de.select_columns import select_columns

class LossPrevTrainingDataPrep(object):
  def __init__(self, redshift_trans_csv_dirname, \
    ml_features_columns_fname, \
    da_features_columns_fname, output_dir):
    """ constructor
    Args:
      redshift_trans_csv_dirname (str): original redshift dump files \
        folder name with all transactions csv files
      ml_features_columns_fname (str): all the features required \
        for ML training. default will be taken from fheaders
      da_features_columns_fname (str): all the features required \
        for data augmentation. Will use these \
        columns to apply rules to generate fraud \
        data for training. default will be taken from fheaders
      output_path (str): absolute path to dump output
    """
    self.redshift_trans_csv_dirname = redshift_trans_csv_dirname
    self.ml_features_columns_fname = ml_features_columns_fname
    self.da_features_columns_fname = da_features_columns_fname
    # Add redshift input dirname as subdir of output_dir
    output_dir = os.path.join(output_dir, \
            os.path.basename(self.redshift_trans_csv_dirname))
    self.output_path = os.path.abspath(output_dir)
    if not os.path.exists(self.output_path):
      os.makedirs(self.output_path)

    # selected columns file name
    self.da_features_select_fname = 'da_select_' + \
      os.path.basename(self.redshift_trans_csv_dirname) + '.csv'
    self.ml_features_select_fname = 'ml_select_' + \
      os.path.basename(self.redshift_trans_csv_dirname) + '.csv'

  def append_labels_to_final_training_data(self, fraud_transactions):
    """ Function appends label to produce final training data
    """
    # Read 'ml_select_<>.csv' file
    fname = os.path.join(self.output_path, \
      self.ml_features_select_fname)

    select_df = pd.read_csv(fname, \
      skip_blank_lines=True, \
      warn_bad_lines=True, error_bad_lines=False)

    def _add_label(row):
      if row['ticketnum'] in fraud_transactions:
        label = 1
      else:
        label = 0
      return label

    # add a column to store label
    tqdm.pandas()
    select_df['label'] = \
        select_df.progress_apply(_add_label, axis=1)
  
    # Dump the file ml_select_with_label_<>.csv file 
    fname = os.path.join(self.output_path, \
      'ml_select_with_label_' + \
      os.path.basename(self.redshift_trans_csv_dirname)) + \
      '.csv'

    select_df.to_csv(fname)

  def feature_engineering(self, select_df):
    """ function to manipulate features 
    Args:
      select_df (DataFrame): transaction data
    Return:
      DataFrame: additional columns based on feature engineering

    1. Ignore redundant transactions
      Focus on below columns only from table:
      locat, dtout (format?), paycode (3,4,6,10), make , \
      color , plate, ccdaccount , ccdexpdate , \
      ratedescription (only pick transactions \
      which are not 'default')

    2. Perform feature engineering
      i. If a card has been used for multiple vehicles, add to \
          _cardmultuses column

    """
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

    # Ignore nan 
    select_df = select_df.dropna(subset=['ccdaccount'])

    # select_df.to_csv(os.path.join(self.output_path, \
        # 'da_select_filtered.csv'))

    ## Perform feature engineering

    # Same card multiple use 
    same_card_mult_usage_dict = {}

    def _build_same_card_multiple_use_map(row):
      """ Same credit card (paycode, ccdaccount, ccdexpdate) \
        used at the same facility (locat) in the \
        same month (dtout) for a different vehicle \
        (make, color, plate)
      """
      same_card_tuple = \
          (row['locat'], row['paycode'], row['ccdaccount'], row['ccdexpdate'])
      if same_card_tuple \
          not in same_card_mult_usage_dict.keys():
        same_card_mult_usage_dict[same_card_tuple] = 1
      else:
        same_card_mult_usage_dict[same_card_tuple] += 1
      
      cardmultuses = same_card_mult_usage_dict[same_card_tuple]
      return cardmultuses

    print('\t\t building same card multiple usage map...')
    select_df['_cardmultuses'] = \
        select_df.apply(_build_same_card_multiple_use_map, axis=1)

    return select_df

  def generate_fraud_training_data(self):
    """ Function generaes fraud data based on rules
    Args:
      None

    Returns:
      None

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

    # Perform feature engineering
    print('\tperforming feature engineering...')
    select_df = self.feature_engineering(select_df)

    # Add fraud label column (if same card was used for multiple vehicles,
    # it's a fraud) - 0 or 1 for no or yes fraud
    fraud_transactions = []
    def _apply_fraud_label(row):
      label = 0
      if row['_cardmultuses'] >= 2:
        # fraud transaction
        label = 1
        fraud_transactions.append(row['ticketnum'])
      return label

    print('\tapplying fraud labels...')
    select_df['label'] = select_df.apply(_apply_fraud_label, axis=1)
    
    # dump the file
    fname = os.path.join(self.output_path, \
      'da_select_filtered_with_label_' + \
      os.path.basename(self.redshift_trans_csv_dirname)) + \
      '.csv'
    select_df.to_csv(fname)

    return fraud_transactions

  def training_data_prep(self):
    """ function to create fradulant training dataset.
    Args:
      None
    Return:
      None

    Description:
      Step-0: Read original redshift dump csv files with all \
        transactions from the folder
      Step-1: Dump file 'da_select_<>.csv' to have all \
        feature columns for data augmentation
      Step-2: Perform feature engineering and using these curated features \
          identify if a transaction is fraud, create labels and dump \
          another file 'da_select_filtered_with_label_<>.csv'
      Step-3: Dump file 'ml_select_<>.csv' to have all \
        feature columns required for ml
      Step-4: Append labels from Step-3 to ml_select_with_label_<>.csv
    """
    # Step-1: 
    # select columns based on self.da_features_columns_fname
    print('\n*** Step-1: selecting da columns...')
    csv_flist = glob.glob(os.path.join(self.redshift_trans_csv_dirname, \
            "*.csv"))
    select_columns(csv_flist, self.da_features_columns_fname, \
      self.output_path, output_fname=self.da_features_select_fname)

    # Step-2:
    # Apply rules to get fraud data
    print('\n*** Step-2: generating fraud training data...')
    fraud_transactions = self.generate_fraud_training_data()

    # Step-3:
    # select columns based on self.ml_features_columns_fname
    print('\n*** Step-3: selecting ml columns...')
    select_columns(csv_flist, self.ml_features_columns_fname, \
      self.output_path, output_fname=self.ml_features_select_fname)

    # Step-4:
    # Augment labels to final training data
    print('\n*** Step-4: augmenting labels to final training data...')
    self.append_labels_to_final_training_data(fraud_transactions)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--redshift_trans_csv_dirname', \
    type=str, required=True, \
    help="Original transaction csv file from redshift dump")
  parser.add_argument('-ml_f', '--ml_features_columns_fname', type=str, \
    default='./fheaders/ml_features_headers.txt', \
    help="Text file to have ML features' headers")
  parser.add_argument('-da_f', '--data_aug_features_columns_fname', \
    type=str, default='./fheaders/training_data_aug_headers.txt', \
    help="Text file to have ML features' headers")
  parser.add_argument('-o', '--output_dir', type=str, \
    required=True, help="Output dir")
  args = parser.parse_args()

  # uncomment to dump keys to keys.txt file
  # dump_keys(trans_df)

  redshift_trans_csv_dirname = args.redshift_trans_csv_dirname
  ml_features_columns_fname = args.ml_features_columns_fname
  da_features_columns_fname = args.data_aug_features_columns_fname

  # Prepare output directory
  output_dir = args.output_dir

  trn_dataprep = LossPrevTrainingDataPrep(redshift_trans_csv_dirname, \
    ml_features_columns_fname, \
    da_features_columns_fname, \
    output_dir)
  trn_dataprep.training_data_prep()


