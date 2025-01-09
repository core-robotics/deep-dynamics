import sys
import csv

file = open(sys.argv[1])
new_datas = []
for line in file:
    data = line.strip().split(',')
    new_data = [float(x) * 0.2 for x in data]
    new_datas.append(new_data)
with open(sys.argv[2], 'w') as f:
    writer = csv.writer(f)
    writer.writerows(new_datas)

file.close()
