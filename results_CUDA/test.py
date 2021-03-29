import numpy as np
import csv


def extract_values(value, csv_file):
    operator = np.ndarray(())
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for (row in csv_reader):
        pass




if __name__ == '__main__':
    with open('sequential_timings.csv', mode = 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(row['operation'])
            if line_count==3:
                number = float(row['time'])
            line_count += 1
        # print(number+1)
        print("completed")


#############################################
########## NOT WOKRING YET  #################
#############################################
