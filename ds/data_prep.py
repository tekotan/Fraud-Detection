import os
import pandas as pd
import numpy as np
import shutil
import multiprocessing
from datetime import datetime
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tqdm import tqdm

from utils import standard_scaler, maxmin_scaler

class FdDataPrep(object):
  def __init__(self, training_dfile, tmp_data_dir='./tmp_train_data_dir'):
    # We would retain all the columns from below list, last one being label.
    self.model_features_list = [
      "locat",
      # "ticketnum",
      "paycode",
      "make",
      "color",
      "plate",
      # "ccdaccount",
      # "ccdexpdate",
      # "ratedescription",
      "label",
    ]

    print("Model Features: {}".format(self.model_features_list))

    self.tmp_data_dir = tmp_data_dir
    if os.path.exists(self.tmp_data_dir) is False:
      os.makedirs(self.tmp_data_dir)
    self.model_features_string_cols = []
    self.model_features_num_cols = []
    self.training_dfile = self.filter_model_features(training_dfile)

  def filter_model_features(self, training_dfile):
    fname = 'model_features_' + os.path.basename(training_dfile)
    new_training_dfile = os.path.join(self.tmp_data_dir, fname)
    train_df = pd.read_csv(training_dfile, \
      skip_blank_lines=True, \
      warn_bad_lines=True, error_bad_lines=False)
    new_train_df = None
    new_train_df = train_df.loc[:, \
        train_df.columns.isin(self.model_features_list)]
    new_train_df.to_csv(new_training_dfile)

    # Get non numerical column idx (strings)
    for idx, value in enumerate(new_train_df.values[0, :]):
      if isinstance(value, float) or isinstance(value, int):
        self.model_features_num_cols.\
            append(new_train_df.columns[idx])
      else:
        self.model_features_string_cols.\
            append(new_train_df.columns[idx])

    return new_training_dfile

  def convert_select_columns_to_numericals(self):
    train_df = pd.read_csv(self.training_dfile)

    # Get non numerical column idx
    string_index = []
    for n, i in enumerate(train_df.values[0, :]):
      if train_df.columns[n] not in self.model_features_list:
        pass
      elif isinstance(i, float) or isinstance(i, int):
        pass
      else:
        string_index.append(n)

    key_dict = {}
    array_df = np.array(train_df.values[1:, :])
    for index in tqdm(range(array_df.shape[1])):
      if train_df.columns[index] not in self.model_features_list:
        pass
      elif index in string_index:
        li = []
        for val in train_df.values[:, index]:
          if val not in li:
            if isinstance(val, str):
              li.append(val)
        key_dict[train_df.keys()[index]] = sorted(li)
    return key_dict
    
  def create_vocab_list_for_string_columns(self):
    train_df = pd.read_csv(self.training_dfile, \
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
    features_df = pd.read_csv(self.training_dfile).dropna(how="any")
    for i in self.model_features_string_cols:
      features_df[i] = features_df[i].astype(str) 
    for i in self.model_features_num_cols:
      features_df[i] = features_df[i].astype(np.float64)
    return features_df[features_df.columns[0:-1]], features_df['label']
