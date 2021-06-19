import os
import csv

res_files = "results_CUDA"
out_folder = "plots/CUDA"


def convert_files(fns:list):
    for fn in fns:
        datas = {"naive":{}, "optimized":{}}
        path = os.path.join(res_files, fn)
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for row in reader:
                if len(row) == 0: continue
                # version,resolution,dilatation_mean,dilatation_max,erosion_mean,erosion_max,opening_mean,opening_max,closing_mean,closing_max
                if row[1] not in datas[row[0]]:
                    datas[row[0]][row[1]] = []
                datas[row[0]][row[1]].append(float(row[2]))

        with open("speedups_cuda_naive.csv", 'w') as file_naive_out:
            spamwriter = csv.writer(file_naive_out, delimiter=',')
            header = ["dataset"] + ["t{}".format(t) for t in tiles]
            spamwriter.writerow(header)
            for k in sorted(datas_naive):
                spamwriter.writerow([k] + datas_naive[k])


if __name__ == '__main__':
    convert_files(["08TileWidthSpeedup.csv","08TileWidthSpeedup.csv","08TileWidthSpeedup.csv","08TileWidthSpeedup.csv"])