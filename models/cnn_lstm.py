import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_same_padding(kernel_size):
    return (kernel_size - 1) // 2

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(5,5), padding=get_same_padding(5))
        self.pool1 = nn.MaxPool2d(kernel_size=(3,1), padding=(get_same_padding(3), 0))
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.dropout1 = nn.Dropout(0.6)

        self.conv2 = nn.Conv2d(25, 50, kernel_size=(5,5), padding=get_same_padding(5))
        self.pool2 = nn.MaxPool2d(kernel_size=(3,1), padding=(get_same_padding(3), 0))
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.dropout2 = nn.Dropout(0.6)

        self.conv3 = nn.Conv2d(50, 100, kernel_size=(5,5), padding=get_same_padding(5))
        self.pool3 = nn.MaxPool2d(kernel_size=(3,1), padding=(get_same_padding(3), 0))
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.dropout3 = nn.Dropout(0.6)

        self.conv4 = nn.Conv2d(100, 200, kernel_size=(5,5), padding=get_same_padding(5))
        self.pool4 = nn.MaxPool2d(kernel_size=(3,1), padding=(get_same_padding(3), 0))
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.dropout4 = nn.Dropout(0.6)

        self.num_flat_features = None

        self.fc = None
        self.lstm = nn.LSTM(40, 10, batch_first=True, dropout=0.4)
        self.fc_final = nn.Linear(10, 4)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)

        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)

        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)

        x = F.elu(self.conv4(x))
        x = self.pool4(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)

        if self.num_flat_features is None:
            self.num_flat_features = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc = nn.Linear(self.num_flat_features, 40).to(x.device)


        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, -1)

        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]

        x = self.fc_final(x)
        return x