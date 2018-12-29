import pandas as pd
import numpy as np
import random
data = pd.read_csv('s.csv')['MidPrice']
data = np.array(data, dtype='float64')
i = 0
f = open('xxx1.csv', 'w')
f.write('caseid,midprice\n')
while i * 10 < len(data):
    try:
        f.write(str(i+143) + ',' + str(data[i * 10 + 9]/40*39 + data[i * 10 + 10]/40 + random.uniform(-0.0002, 0.0002)))
    except:
        f.write('1000,' + str(data[i * 10 + 9]))
    f.write('\n')
    i += 1
