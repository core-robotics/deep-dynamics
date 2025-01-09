import sys
import csv

datas = []
with open(sys.argv[1]) as f:
    for line in f:
        data = line.strip().split(',')
        del data[2::3]
        datas.append(data)
with open(sys.argv[2], 'w') as f:
    writer = csv.writer(f)
    writer.writerows(datas)
