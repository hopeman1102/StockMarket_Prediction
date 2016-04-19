#! /usr/bin/python
'''
    Data Interpolation
'''

import os, sys
import pandas as pd

def interpolate(dataframe, cols_to_interpolate):

    for col in cols_to_interpolate:
        dataframe[col] = dataframe[col].interpolate('spline', order=2)

    return dataframe


def main(dir_path, output_dir):
    files = os.listdir(dir_path)
    for file_name in files:
        dataframe = pd.read_csv(os.path.join(dir_path, file_name))
        dataframe = interpolate(dataframe, \
            ['high', 'open', 'high', 'low', 'close', 'volume', 'adj_close'])
        print dataframe

        break


if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])
