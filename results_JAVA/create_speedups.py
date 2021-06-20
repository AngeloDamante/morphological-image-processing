import os
import pandas as pd


def create_speedups(fn: str, debug=False):
    print(fn)
    df = pd.read_csv(fn)
    if debug: print(df.columns)
    for col in df.columns[2:]:
        df[col] = df['1'] / df[col]
    if debug: print(df.head())
    new_name = "speedups"+ fn[len(fn.split('_')[0]):]
    df.drop('1', axis=1, inplace=True)
    df.set_index("Resolution")
    df.to_csv(new_name, index=False)


if __name__ == '__main__':
    create_speedups("timings_java_dilation_optimized.csv")
    create_speedups("timings_java_erosion_optimized.csv")
