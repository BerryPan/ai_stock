import pandas as pd
import numpy as np

data = pd.read_csv('lalala.csv')
i = 0
midprice = np.array(data[['MidPrice']])

target_f = open('last_time.csv', 'w')
target_f.write('MidPrice\n')
print(data.shape)
mid = []
while i * 10 < data.shape[0]:
    mid.append(midprice[i*10+9][0])
    target_f.write(str(midprice[i*10+9][0]) + '\n')
    i += 1
