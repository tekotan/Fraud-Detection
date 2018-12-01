import os
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import shutil
import multiprocessing
from datetime import datetime
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tqdm import tqdm

from utils import standard_scaler, maxmin_scaler

class FdDataPrep(object):
  def __init__(self, datafile, tmp_data_dir='./tmp_train_data_dir'):
    # We would retain all the columns from below list, last one being label.
    self.model_features_list = [
      # 'locat',
      # 'ticketnum',
      # 'paycode',
      # 'make',
      # 'color',
      # 'plate',
      # 'ccdaccount',
      # 'ccdexpdate',
      # 'ratedescription',
      '_cardmultuses',
      'label',
    ]

    print('Model Features: {}'.format(self.model_features_list))

    self.tmp_data_dir = tmp_data_dir
    if os.path.exists(self.tmp_data_dir) is False:
      os.makedirs(self.tmp_data_dir)
    self.model_features_string_cols = []
    self.model_features_num_cols = []
    self.datafile = self.filter_model_features(datafile)

  def filter_model_features(self, datafile):
    """ From input data file, filter features
        from model_features_list

      Args:
        datafile (str): input data file
      Return:
        str: file name with new training data
      Description:
        This function first filters input data file to new training data file
        Secondly it finds numerical cols and string cols and populate for 
          later use
    """
    fname = 'model_features_' + os.path.basename(datafile)
    new_training_dfile = os.path.join(self.tmp_data_dir, fname)
    train_df = pd.read_csv(datafile, \
      skip_blank_lines=True, \
      warn_bad_lines=True, error_bad_lines=False)
    new_train_df = None
    new_train_df = train_df.loc[:, \
        train_df.columns.isin(self.model_features_list)]
    new_train_df.to_csv(new_training_dfile)

    # Get non numerical column idx (strings)
    for col in new_train_df.columns:
      if is_numeric_dtype(new_train_df[col]):
        self.model_features_num_cols.append(col)
      elif is_string_dtype(new_train_df[col]):
        self.model_features_string_cols.append(col)

    return new_training_dfile

  def create_vocab_list_for_string_columns(self):
    train_df = pd.read_csv(self.datafile, \
      skip_blank_lines=True, \
      warn_bad_lines=True, error_bad_lines=False)
    string_keys = {}
    for col in self.model_features_string_cols:
      value = sorted(train_df[col].astype(str).unique())
      string_keys[col] = value
    return string_keys

  def get_feature_columns(self):
    string_col_keys = self.create_vocab_list_for_string_columns()

    feature_columns = {}

    for index in self.model_features_list[:-1]:
      if index in string_col_keys.keys():
        if len(string_col_keys[index]) < 100:
          feature_columns[index] = tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
              index, vocabulary_list=string_col_keys[index]
            )
          )
        else:
          feature_columns[index] = tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
              index, hash_bucket_size=100, dtype=tf.string
            )
          )
      else:
        feature_columns[index] = tf.feature_column.numeric_column(
          index)
    print("Features Columns \n {}".format(feature_columns))
    return feature_columns

  def read_training_data(self):
    features_df = pd.read_csv(self.datafile).dropna(how="any")
    for i in self.model_features_string_cols:
      features_df[i] = features_df[i].astype(str) 
    for i in self.model_features_num_cols:
      features_df[i] = features_df[i].astype(np.float64)
    return features_df[features_df.columns[1:-1]], features_df['label']

  def read_predict_data(self, fname):
    features_df = pd.read_csv(fname).dropna(how="any")
    features_df = features_df.astype(np.int64)
    return features_df[features_df.columns[1:]]
