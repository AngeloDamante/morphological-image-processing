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
    _type, _framework, _version, _operation = fn[:-4].split('_')
    df = pd.read_csv(path)
    df.set_index("dataset" if _framework=="cuda" else "Resolution", inplace=True)
    if debug: print(df.head())
    if transpose: df = df.transpose()
    if debug: print(df.head())
    if debug: print(df.columns)
    ax = df.plot(title="{} for {} ({} version) on {} operation".format(_type, _framework, _version, _operation), marker='.', markersize=10, figsize=(12, 7))
    x_label = ("Tile Width" if _framework == "cuda" else "Threads") if transpose else "dataset size"
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
    src_folder = 'results_CUDA'
    out_folder = 'plots/CUDA'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for tr in [True, False]:
        for file in os.listdir(src_folder):
            if '_' in file and file.endswith('.csv'):
                create_result_plot(file, tr)

    src_folder = 'results_JAVA'
    out_folder = 'plots/JAVA'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for tr in [True, False]:
        for file in os.listdir(src_folder):
            if '_' in file and file.endswith('.csv'):
                create_result_plot(file, tr)