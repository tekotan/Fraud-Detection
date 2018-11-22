import numpy as np
import pandas as pd
from tqdm import tqdm
import math


def convert_select_columns_to_numericals(select_csv_fname):
    df = pd.read_csv(select_csv_fname)
    array_df = np.array(df.values()[1:, :])
    key_dict = {}
    string_index = []
    fin_list = []
    df = df.dropnan(how="any")
    for n, i in enumerate(df.values[0, :]):
        if isinstance(i, float) or isinstance(i, int):
            pass
        else:
            string_index.append(n)
    for index in tqdm(range(array_df.shape[1])):
        if index in string_index:
            li = []
            for val in df.values[:, index]:
                if val not in li:
                    if isinstance(val, str):
                        li.append(val)
            key_dict["key_" + df.keys()[index]] = sorted(li
            for n, i in enumerate(array_df[:, index]):
                if isinstance(i, str):
                    temp = np.zeros(len(key_dict["key_" + df.keys()[index]]))
                    temp[key_dict["key_" + df.keys()[index]].index(i)] = 1
                    array_df[n, index] = temp
                else:
                    temp = np.zeros(len(key_dict["key_" + df.keys()[index]]))
                    array_df[n, index] = temp
        fin_list.append(np.vstack(array_df[1:, index]))
    print(type(array_df))
    for i in range(len(fin_list)):
        fin_list[i] = np.vstack(fin_list[i])
    import ipdb
    ipdb.set_trace
    fin_array = np.vstack(fin_list)
    return fin_array, key_dict
