import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

EPOCH = 1
BATCH = 1
TIME_STEP = 1
INPUT_SIZE = 4
LR = 0.01


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = BATCH
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=self.hidden_size,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,
        )
        self.out = nn.Linear(BATCH, 1)

    def forward(self, x):
        self.hidden = (torch.zeros(1, BATCH, self.hidden_size),
                       torch.zeros(1, BATCH, self.hidden_size))
        r_out, self.hidden = self.rnn(x, self.hidden)
        out = self.out(r_out[:, -1, :])
        return out


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        train = pd.read_csv(filepath)
        # train = pd.read_csv('test_target.csv')
        # target_array = np.array(train[['MidPrice']])
        # # target_array = preprocessing.scale(target_array)
        # target = torch.tensor(target_array.astype(np.float32))
        # train = pd.read_csv('test.csv')
        target = train["MidPrice"]-train["MidPrice"].shift(1)
        target = target.fillna(0)
        target = np.array(target).astype(np.float32)
        target = torch.tensor(target)
        volume = train['BidVolume1'] - train['AskVolume1']
        other = train[['AskPrice1', 'BidPrice1', 'Volume']]
        volume = pd.DataFrame({'MidVolume': list(volume)})
        data = pd.concat([other, volume], axis=1)
        data = np.array(data)
        for i in range(INPUT_SIZE):
            data[:, i] = preprocessing.scale(data[:, i])
        data = torch.tensor(data.astype(np.float32))
        self.len = data.shape[0]
        self.x_data = data
        self.y_data = target

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


test_loss = 0
loss_func = nn.MSELoss(size_average=False)
rnn = torch.load('net.pkl')

test_data = DiabetesDataset(filepath='s.csv')
test_loader = DataLoader(dataset=test_data, batch_size=BATCH, shuffle=False)
f = open('123.csv', 'w')
f.write('caseid,midprice\n')
last_time = np.array(pd.read_csv('last_time.csv'))
for step, (data, target) in enumerate(test_loader):  # gives batch data
    data, target = Variable(data), Variable(target)
    data = data.view(-1, TIME_STEP, INPUT_SIZE)
    output = rnn(data)
    loss = loss_func(output, target)
    test_loss += loss_func(output, target).item()
    if step%10 == 9:
        f.write(str(step//10+143)+','+str(last_time[step//10][0]+output.item())+'\n')
test_loss /= len(test_loader.dataset)
print('\nTest set:Average Loss:{:.6f}\n'.format(test_loss))

