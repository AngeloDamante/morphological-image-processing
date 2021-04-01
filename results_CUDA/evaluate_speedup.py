import numpy as np
import os
import csv


def extract_values(results_file, resolution):
    dilatation, erosion, closing, opening = [], [], [], []

    with open(results_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for row in csv_reader:
            image_resolution = row['image'][8:8+len(resolution)]

            if (image_resolution == resolution and row['operation'] == 'Dilatation'):
                dilatation.append(float(row['time']))
            elif (image_resolution == resolution and row['operation'] == 'Erosion'):
                erosion.append(float(row['time']))
            elif (image_resolution == resolution and row['operation'] == 'Opening'):
                opening.append(float(row['time']))
            elif (image_resolution == resolution and row['operation'] == 'Closing'):
                closing.append(float(row['time']))
        # end
    # end

    timings = np.zeros((len(dilatation), 4))
    timings[:, 0] = dilatation
    timings[:, 1] = erosion
    timings[:, 2] = closing
    timings[:, 3] = opening

    return timings
# end


def compute_speedup(sequential, parallel, resolution):
    print(f"Compute speedup for {parallel['version']} version with {resolution} resolution")

    # ndarray with 4 MM operations
    seq_timing = extract_values(sequential['file'], resolution)
    par_timing = extract_values(parallel['file'], resolution)

    # results
    speedup = np.divide(seq_timing, par_timing)
    mean_speedup = speedup.mean(axis=0)
    max_speedup = np.max(speedup, axis=0)

    # results mining
    result = {'version': parallel['version'], 'resolution':resolution,
              'dilatation_mean':mean_speedup[0], 'dilatation_max':max_speedup[0],
              'erosion_mean':mean_speedup[1], 'erosion_max':max_speedup[1],
              'opening_mean':mean_speedup[2], 'opening_max':max_speedup[2],
              'closing_mean':mean_speedup[3], 'closing_max':max_speedup[3]
             }

    return result
# end


if __name__ == '__main__':
    tile_width = os.getenv('NUM_THREAD_PER_AXIS')
    print(f'Speedup with TILE_WIDTH = {tile_width} evaluating...')

    resolution = ['1280x720', '1920x1080', '2560x1440', '3840x2160', '7680x4320']

    # csv files
    seq = {'version':'sequential', 'file':'sequential_timings.csv'}
    naive = {'version':'naive', 'file':'naive_timing.csv'}
    optimized = {'version':'optimized', 'file':'optimized_timing.csv'}

    # fields for results file
    basic = ['dilatation_mean', 'dilatation_max', 'erosion_mean', 'erosion_max']
    composed = ['opening_mean', 'opening_max', 'closing_mean', 'closing_max']

    # write on csv
    output_file = str(tile_width) + "TileWidthSpeedup.csv"
    with open(output_file, 'w', newline='') as file:
        fieldnames = ['version', 'resolution'] + basic + composed
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for res in resolution:
            writer.writerow(compute_speedup(seq, naive, res))
            writer.writerow(compute_speedup(seq, optimized, res))
        # end

    # end

    print("___file-written___")
# end
