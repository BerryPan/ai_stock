import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

EPOCH = 1
BATCH = 1
TIME_STEP = 10
INPUT_SIZE = 5
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
        target_array = np.array(train[['MidPrice']])
        # target_array = preprocessing.scale(target_array)
        target = torch.tensor(target_array.astype(np.float32))
        data = train[['AskPrice1', 'BidPrice1', 'Volume', 'BidVolume1', 'AskVolume1']]
        data = np.array(data)
        for i in range(INPUT_SIZE):
            data[:, i] = preprocessing.scale(data[:, i])
        data = torch.tensor(data.astype(np.float32))
        self.len = data.shape[0]
        self.x_data = data
        self.y_data = target
        print(self.y_data.shape)

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
for step, (data, target) in enumerate(test_loader):  # gives batch data
    if step%10 == 9:
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 1, INPUT_SIZE)
        output = rnn(data)
        loss = loss_func(output, target)
        print(output)
        print(target)
        print(loss)
        test_loss += loss_func(output, target).item()
        predicted = torch.max(output, 1)[0].data.numpy()
        print(predicted)
        f.write(str(step+143)+','+str(predicted[-1])+'\n')
test_loss /= len(test_loader.dataset)
print('\nTest set:Average Loss:{:.6f}\n'.format(test_loss))

