import pandas as pd
import numpy as np

data = pd.read_csv('train_data.csv')
i = 0
midprice = np.array(data[['MidPrice']])

f = open('train.csv', 'w')
f.write('Volume,BidPrice1,BidVolume1,AskPrice1,AskVolume1\n')
target_f = open('midprice.csv', 'w')
target_f.write('MidPrice\n')
input_data = []
tag = ['Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
for item in tag:
    input_data.append(np.array(data[[item]]))
print(data.shape)
print(input_data[0].shape)
while i * 30 < data.shape[0]:
    target_f.write(str(midprice[i * 30 + 10:i * 30 + 29].mean()) + '\n')
    for j in range(5):
        f.write(str(input_data[j][i * 30:i * 30 + 9].mean()))
        if j == 4:
            f.write('\n')
        else:
            f.write(',')

    i += 1
