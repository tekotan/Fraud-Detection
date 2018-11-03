import pandas as pd
import numpy as np
import shutil
import multiprocessing
from datetime import datetime

import tensorflow as tf
from tensorflow.python.feature_column import feature_column

from utils import standard_scaler, maxmin_scaler

class FdDataPrep(object):
  def __init__(self):
    FEATURE_COUNT = 64
    self.MULTI_THREADING = True

    self.HEADER = ['key']
    self.HEADER_DEFAULTS = [[0]]
    self.UNUSED_FEATURE_NAMES = ['key']
    self.LABEL_FEATURE_NAME = 'LABEL'
    self.FEATURE_NAMES = []  

    for i in range(FEATURE_COUNT):
      self.HEADER += ['x_{}'.format(str(i+1))]
      self.FEATURE_NAMES += ['x_{}'.format(str(i+1))]
      self.HEADER_DEFAULTS += [[0.0]]

    self.HEADER += [self.LABEL_FEATURE_NAME]
    self.HEADER_DEFAULTS += [['NA']]

    print("self.Header: {}".format(self.HEADER))
    print("Features: {}".format(self.FEATURE_NAMES))
    print("Label Feature: {}".format(self.LABEL_FEATURE_NAME))
    print("Unused Features: {}".format(self.UNUSED_FEATURE_NAMES))

  def parse_csv_row(self, csv_row):
    
    columns = tf.decode_csv(csv_row, record_defaults=self.HEADER_DEFAULTS)
    features = dict(zip(self.HEADER, columns))
    
    for column in self.UNUSED_FEATURE_NAMES:
      features.pop(column)

    target = features.pop(self.LABEL_FEATURE_NAME)

    return features, target

  def csv_input_fn(self, files_name_pattern, mode=tf.estimator.ModeKeys.EVAL, 
           skip_header_lines=0, 
           num_epochs=None, 
           batch_size=200):
    
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    
    print("")
    print("* data input_fn:")
    print("================")
    print("Input file(s): {}".format(files_name_pattern))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")
    
    file_names = tf.matching_files(files_name_pattern)

    dataset = tf.data.TextLineDataset(filenames=file_names)
    dataset = dataset.skip(skip_header_lines)
    
    if shuffle:
      dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
      
    num_threads = multiprocessing.cpu_count() if self.MULTI_THREADING else 1
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda csv_row: self.parse_csv_row(csv_row), num_parallel_calls=num_threads)
    
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    
    features, target = iterator.get_next()

    return features, target

  def get_feature_columns(self):
    """
    df_params = pd.read_csv("data/params.csv", header=0, index_col=0)
    len(df_params)
    df_params['feature_name'] = self.FEATURE_NAMES
    df_params.head()
    """

    feature_columns = {}

  #   feature_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
  #            for feature_name in self.FEATURE_NAMES}

    for feature_name in self.FEATURE_NAMES:

      feature_max = 100000 #df_params[df_params.feature_name == feature_name]['max'].values[0]
      feature_min = -100000 #df_params[df_params.feature_name == feature_name]['min'].values[0]
      normalizer_fn = lambda x: maxmin_scaler(x, feature_max, feature_min)
      
      feature_columns[feature_name] = tf.feature_column.numeric_column(feature_name, 
                                       # Disable normalizer
                                       # normalizer_fn=normalizer_fn
                                      )
   

    return feature_columns


