import sys
import csv
import numpy as np

datas = []
with open(sys.argv[1]) as f:
    for line in f:
        data = line.strip().split(',')
        datas.append(data)
data_numpy = np.asarray(datas)
with open(sys.argv[2], 'w') as f:
    f.write(','.join(data for data in data_numpy[:,0]))
    f.write("\n")
    f.write(','.join(data for data in data_numpy[:,1]))