""" Create trainig data based on rules for ML training """
import argparse
import os
from de.select_columns import select_columns
from de.convert_fields_to_numericals import convert_select_columns_to_numericals

class LossPrevTrainingDataPrep(object):
    def __init__(self, redshift_trans_csv_fname, \
            ml_features_columns_fname, output_dir):
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
        self.da_features_columns_fname = ds_features_fname
        self.output_path = os.path.abspath(output_dir)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # selected columns file name
        self.da_features_select_name = 'da_select_' + \
                os.path.basename(self.redshift_trans_csv_fname)
        self.ml_features_select_name = 'ml_select_' + \
                os.path.basename(self.redshift_trans_csv_fname)

    def append_labels_to_final_training_data(self):
        """ Function appends label to produce final training data
        """
        # Read ml_select_<>.csv, match ticketnum ID and add label to \
                # ml_select_with_label_<>.csv

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
        # Apply above rules to generate label
        # Append it to da_select_with_label_<>.csv file and dump
        

    def training_data_prep(self)
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
                output_fname=self.da_features_select_name)

        # Step-2:
        # Apply rules to get fraud data
        self.generate_fraud_training_data()

        # Step-3:
        # select columns based on self.ml_features_columns_fname
        select_columns(self.redshift_trans_csv_fname, \
                self.ml_features_columns_fname, \
                self.output_path, \
                output_fname=self.ml_features_select_name)

        # Step-4:
        # Augment labels to final trainig data
        self.append_labels_to_final_training_data()

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


