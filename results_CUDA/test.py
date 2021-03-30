import numpy as np
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

## TODO: NOT YET FIISHED
if __name__ == '__main__':
    seq1, seq2, seq3, seq4, seq5 = extract_results('sequential_timings.csv')
    nv1, nv2, nv3, nv4, nv5 = extract_results('naive_timing.csv')
    opt1, opt2, opt3, opt4, opt5 = extract_results('optimized_timing.csv')

    speedup1280x720 = compute_speedup(seq5, opt5)
    # print(speedup1280x720)
    print(seq4, opt4)

    pass
# end
