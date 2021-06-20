import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

src_folder = 'results_CUDA'
out_folder = 'plots/CUDA'
show_plots=False

def create_result_plot(fn:str, transpose=True, debug=False):
    path = os.path.join(src_folder, fn)
    if not os.path.exists(path):
        print("File does not exists: {}".format(fn))
        return
    if debug: print()
    print(fn)
    df = pd.read_csv(path)
    df.set_index("dataset", inplace=True)
    if debug: print(df.head())
    if transpose: df = df.transpose()
    if debug: print(df.head())
    if debug: print(df.columns)
    _type, _framework, _version, _operation = fn[:-4].split('_')
    ax = df.plot(title="{} for {} ({} version) on {} operation".format(_type, _framework, _version, _operation), marker='.', markersize=10, figsize=(12, 7))
    x_label = ("Tile Width" if _framework == "cuda" else "Threads") if not transpose else "dataset size"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time (seconds)" if _type == "timings" else "Speedup value")
    if show_plots: plt.show()
    fig = ax.get_figure()
    fig.savefig(os.path.join(out_folder, fn[:-4]+"{}.jpg".format("_by_dataset" if transpose else "_by_t")))

def create_plots():
    for file in os.listdir(src_folder):
        if file.endswith('.csv'):
            print()
            print(file)
            file_path = os.path.join(src_folder, file)
            df = pd.read_csv(file_path)
            print(df.head())
            df.transpose()[1:].plot(title=file[:-4])
            plt.show()


if __name__ == '__main__':
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    #create_plots()
    for tr in [True, False]:
        for file in os.listdir(src_folder):
            if '_' in file and file.endswith('.csv'):# filter only files created with convert script
                create_result_plot(file, tr)
        #create_result_plot("speedups_cuda_shared.csv", tr)
        #àcreate_result_plot("speedups_java_dynamic.csv", tr)
        #create_result_plot("speedups_java_static.csv", tr)