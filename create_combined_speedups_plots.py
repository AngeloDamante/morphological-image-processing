import os
import pandas as pd
import matplotlib.pyplot as plt

out_folder = "plots/CUDA"
show_plots = False


def create_combined(transpose:bool, debug=False):
    pd_1 = pd.read_csv("results_CUDA/speedups_cuda_naive_dilatation.csv")
    pd_2 = pd.read_csv("results_CUDA/speedups_cuda_optimized_dilatation.csv")

    df = pd.DataFrame(pd_1["dataset"])
    df["naive"] = pd_1["t24"]
    df["optimized"] = pd_2["t24"]

    df.set_index("dataset", inplace=True)
    if debug: print(df.head())
    if transpose: df = df.transpose()
    if debug: print(df.head())
    if debug: print(df.columns)

    ax = df.plot(title="Naive vs Optimized on TW=24", marker='.', markersize=10, figsize=(12, 7))
    x_label = ("Tile Width") if transpose else "dataset size"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Speedup value")
    if show_plots: plt.show()
    fig = ax.get_figure()
    fig.savefig(os.path.join(out_folder, "speedups_cuda_combined{}.jpg".format("_by_dataset" if transpose else "_by_t")))


if __name__ == '__main__':
    create_combined(transpose=True)
    create_combined(transpose=False)