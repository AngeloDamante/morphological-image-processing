import os
import csv

src_folder = "TMP"


def convert_files(fns: list):
    datas = {"naive": {}, "optimized": {}}
    operations = ["dilatation", "erosion", "opening", "closure"]
    for k, v in datas.items():
        for o in operations:
            v[o] = {}
    tiles = []

    def reaad_one_tile(fn, datas, tiles):
        tile_n = int(os.path.basename(fn)[0:2])
        tiles.append(tile_n)
        with open(fn) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for row in reader:
                if len(row) == 0: continue
                version, resolution, dilatation_mean, dilatation_max, erosion_mean, erosion_max, opening_mean, opening_max, closing_mean, closing_max = row
                for i, op in enumerate(operations):
                    if resolution not in datas[version][op]:
                        datas[version][op][resolution] = []
                    datas[version][op][resolution].append(float(row[2 + i * 2].replace(',', '.')))

    for f in sorted(fns):
        reaad_one_tile(f, datas, tiles)

    for ver, d in datas.items():
        for op, datas_res in d.items():
            out_fn = "speedups_cuda_{}_{}.csv".format(ver, op)
            print(out_fn)
            with open(out_fn, 'w') as file_naive_out:
                spamwriter = csv.writer(file_naive_out, delimiter=',')
                header = ["dataset"] + ["t{}".format(t) for t in tiles]
                spamwriter.writerow(header)
                for k in sorted(datas_res):
                    spamwriter.writerow([str(k)] + datas_res[k])


if __name__ == '__main__':
    files = []
    for f in os.listdir(src_folder):
        files.append(os.path.join(src_folder, f))
    convert_files(files)
