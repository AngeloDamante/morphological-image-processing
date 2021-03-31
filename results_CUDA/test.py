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


def extract_results(results_file):
    res1280x720 = extract_values(results_file, '1280x720')
    res1920x1080 = extract_values(results_file, '1920x1080')
    res2560x1440 = extract_values(results_file, '2560x1440')
    res3840x2160 = extract_values(results_file, '3840x2160')
    res7680x4320 = extract_values(results_file, '7680x4320')

    return res1280x720, res1920x1080, res2560x1440, res3840x2160, res7680x4320
# end

def compute_speedup(seq_timing, par_timing):
    return np.divide(seq_timing, par_timing)
#end

def extract_speedups(tile_width, seq_version, par_version):
    seq1, seq2, seq3, seq4, seq5 = extract_results(seq_version)
    par1, par2, par3, par4, par5 = extract_results(par_version)

    speedup1280x720 = compute_speedup(seq1, par1)
    speedup1920x1080 = compute_speedup(seq2, par2)
    speedup2560x1440 = compute_speedup(seq3, par3)
    speedup3840x2160 = compute_speedup(seq4, par4)
    speedup7680x4320 = compute_speedup(seq5, par5)

    # TODO: compute mean!
    # TODO: write on csv!
    pass
#end

if __name__ == '__main__':

    tile_width = os.getenv('NUM_THREAD_PER_AXIS')

    sequential_version = 'sequential_timings.csv'
    naive_version = 'naive_timing.csv'
    optimized_version = 'optimized_timing.csv'

    extract_speedups(tile_width, sequential_version, naive_version)
    extract_speedups(tile_width, sequential_version, optimized_version)

    print("___completed___")

# end
