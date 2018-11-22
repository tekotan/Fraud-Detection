import pandas as pd
import numpy as np
import shutil
import multiprocessing
from datetime import datetime
from convert_fields_to_numericals import convert_select_columns_to_numericals
import tensorflow as tf
from tensorflow.python.feature_column import feature_column

from utils import standard_scaler, maxmin_scaler


class FdDataPrep(object):
    def __init__(self):
        FEATURE_COUNT = 11
        self.MULTI_THREADING = True

        self.HEADER = ["key"]
        self.HEADER_DEFAULTS = [[0]]
        self.UNUSED_FEATURE_NAMES = ["key"]
        self.LABEL_FEATURE_NAME = "LABEL"
        self.FEATURE_NAMES = [
            "locat",
            "ticketnum",
            "paycode",
            "make",
            "color",
            "plate",
            "ccdaccount",
            "ccdexpdate",
            "ratedescription",
            "label",
        ]

        #        for i in range(FEATURE_COUNT):
        #            self.HEADER += ["x_{}".format(str(i + 1))]
        #            self.FEATURE_NAMES += ["x_{}".format(str(i + 1))]
        #            self.HEADER_DEFAULTS += [[0.0]]

        self.HEADER += [self.LABEL_FEATURE_NAME]
        self.HEADER_DEFAULTS += [["NA"]]

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

    def get_feature_columns(self):
        """
    df_params = pd.read_csv("data/params.csv", header=0, index_col=0)
    len(df_params)
    df_params['feature_name'] = self.FEATURE_NAMES
    df_params.head()
    """
        TRAIN_DATA_FILE = "../de/trn_data_out/da_select_with_label_Orig_Data.csv"
        key = convert_select_columns_to_numericals(TRAIN_DATA_FILE)
        feature_columns = {}

        #   feature_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
        #            for feature_name in self.FEATURE_NAMES}
        # s,0,locat,ticketnum,dtout,paycode,make,color,plate,ccdaccount,ccdexpdate,ratedescription,label
        # 9,9,705.0,256828.0,2018-06-04 14:48:32.817,4.0,Honda,White,BKD6171,1751,0420,Joint Diseases Hospital,0
        # for feature_name in self.FEATURE_NAMES:
        #
        #     feature_max = (
        #         100000
        #     )  # df_params[df_params.feature_name == feature_name]['max'].values[0]
        #     feature_min = (
        #         -100000
        #     )  # df_params[df_params.feature_name == feature_name]['min'].values[0]
        #     # normalizer_fn = lambda x: maxmin_scaler(x, feature_max, feature_min)
        #
        #     feature_columns[feature_name] = tf.feature_column.numeric_column(
        #         feature_name,
        #         # Disable normalizer
        #         # normalizer_fn=normalizer_fn
        #     )
        for index in self.FEATURE_NAMES:
            if index in key.keys():
                if len(key[index]) < 100:
                    feature_columns[index] = tf.feature_column.indicator_column(
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            index, vocabulary_list=key[index]
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
