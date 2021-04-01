__author__ = 'Angelo D.Amante'
__mail__ = 'angelo.damante16@gmail.com'

import numpy as np
import os
import csv


def extract_values(results_file, resolution):
    '''
    Extracts timings for each MM operation from csv file with chosen resolution.

        Parameters:
            results_file (str): name of csv file
            resolution (str): for image 'widthxheight'

        Returns:
            timings (ndarray): Dilatation | Erosion | Opening | CLosing
                there are more measurements to average out.
    '''
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
    '''
    Extracts mean and max speedup for chosen resolution.

        Parameters:
            sequential (dict):
                - version (field str): version of sequential version
                - file (field str): name of csv file
            parallel (dict):
                - version (field str): version of parallel version
                - file (field str): name of csv file
            resolution (str): widthxheight

        Returns:
            result (dict):
                - version (field str): version of parallel computed
                - resolution (field str): chosen resolution
                - dilatation_mean (field float): mean value for dilatation op
                - dilatation_max (field float): max value for dilatation op
                - erosion_mean (field float): mean value for erosion op
                - erosion_max (field float): max value for erosion op
                - opening_mean (field float): mean value for opening op
                - opening_max (field float): max value for opening op
                - closing_mean (field float): mean value for closing op
                - closing_max (field float): max value for closing op
    '''
    print(f"Compute speedup for {parallel['version']} version with {resolution} resolution")

    # ndarray with 4 MM operations
    seq_timing = extract_values(sequential['file'], resolution)
    par_timing = extract_values(parallel['file'], resolution)

    # results
    speedup = np.divide(seq_timing, par_timing)
    mean_speedup = speedup.mean(axis=0)
    max_speedup = np.max(speedup, axis=0)

    # results mining
    result = {'version': parallel['version'], 'resolution': resolution,
              'dilatation_mean': mean_speedup[0], 'dilatation_max': max_speedup[0],
              'erosion_mean': mean_speedup[1], 'erosion_max': max_speedup[1],
              'opening_mean': mean_speedup[2], 'opening_max': max_speedup[2],
              'closing_mean': mean_speedup[3], 'closing_max': max_speedup[3]
              }

    return result
# end


if __name__ == '__main__':
    tile_width = os.getenv('NUM_THREAD_PER_AXIS')
    print(f'Speedup with TILE_WIDTH = {tile_width} evaluating...')

    resolution = ['1280x720', '1920x1080', '2560x1440', '3840x2160', '7680x4320']

    # csv files
    seq = {'version': 'sequential', 'file': 'sequential_timings.csv'}
    naive = {'version': 'naive', 'file': 'naive_timing.csv'}
    optimized = {'version': 'optimized', 'file': 'optimized_timing.csv'}

    # fields for results file
    basic = ['dilatation_mean', 'dilatation_max', 'erosion_mean', 'erosion_max']
    composed = ['opening_mean', 'opening_max', 'closing_mean', 'closing_max']

    # write on csv
    output_file = str(tile_width) + "TileWidthSpeedup.csv"
    output_file = output_file if tile_width != None else "Speedup.csv"
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
