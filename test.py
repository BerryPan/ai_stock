import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing


EPOCH = 1
BATCH = 10
TIME_STEP = 10
INPUT_SIZE = 4
LR = 0.01


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # self.hidden_size = 10
        self.rnn = nn.LSTM(
            input_size=4,
            hidden_size=8,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,
        )
        # self.hidden = (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
        #                torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)))
        self.out = nn.Linear(8, 10)

    def forward(self, x):
        r_out, self.hidden= self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        train = pd.read_csv(filepath).head(10)

        volume = train['BidVolume1'] - train['AskVolume1']
        target_array = np.array(train[['MidPrice']])
        # target_array = preprocessing.scale(target_array)
        target = torch.tensor(target_array.astype(np.float32))
        print(target)
        other = train[['AskPrice1', 'BidPrice1', 'Volume']]
        volume = pd.DataFrame({'MidVolume': list(volume)})
        data = pd.concat([other, volume], axis=1 )
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
for step, (data, target) in enumerate(test_loader):  # gives batch data
    data, target = Variable(data), Variable(target)
    data = data.view(-1, 1, 4)
    output = rnn(data)
    loss = loss_func(output, target)
    test_loss += loss_func(output, target).item()
    predicted = torch.max(output, 1)[0].data.numpy()
    print(predicted)
test_loss /= len(test_loader.dataset)
print('\nTest set:Average Loss:{:.6f}\n'.format(test_loss))
print(predicted)
