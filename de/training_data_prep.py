import argparse
import os
from de.select_columns import select_columns
from de.convert_fields_to_numericals import convert_select_columns_to_numericals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_csv_file', type=str, \
            required=True, help="Input csv file")
    parser.add_argument('-f', '--feature_columns', type=str, \
            required=True, help="Text file to have features (headers)")
    parser.add_argument('-o', '--output_dir', type=str, \
            required=True, help="Output dir")
    args = parser.parse_args()

    # uncomment to dump keys to keys.txt file
    # dump_keys(trans_df)

    trans_fname = args.input_csv_file
    features_fname = args.feature_columns
    output_dir = args.output_dir
    output_path = os.path.abspath(output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # select columns based on features_fname
    # select_columns(trans_fname, features_fname, output_path)
    # convert strings from selected columns to numericals
    select_columns_fname = os.path.join(output_path, 'select_'+os.path.basename(trans_fname))
    import ipdb; ipdb.set_trace()
    data_value_array, keys = convert_select_columns_to_numericals(select_columns_fname)
