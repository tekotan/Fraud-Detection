import numpy as np
import pandas as pd
from tqdm import tqdm
import math

def convert_select_columns_to_numericals(select_csv_fname):
    df = pd.read_csv(select_csv_fname)
    array_df = np.array(df)
    key_dict = {}
    string_index=[]
    for n, i in enumerate(df.values[0, :]):
        if isinstance(i, float) or isinstance(i, int):
            pass
        else:
            string_index.append(n)
    for index in tqdm(string_index):
        li = []
        for val in df.values[:, index]:
            if val not in li:
                if isinstance(val, str):
                    li.append(val)
        key_dict["key_"+df.keys()[index]] = sorted(li)
        for n, i in enumerate(array_df[:, index]):
            if isinstance(i, str):
                temp = np.zeros(len(key_dict["key_"+df.keys()[index]]))
                temp[key_dict["key_"+df.keys()[index]].index(i)] = 1
                array_df[n, index] = temp
            else:
                temp = np.zeros(len(key_dict["key_"+df.keys()[index]]))
                array_df[n, index] = temp 
    return array_df, key_dict
