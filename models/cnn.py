import torch
import torch.nn as nn
import torch.nn.functional as F

def get_same_padding(kernel_size):
    return (kernel_size - 1) // 2

class CNN(nn.Module):
    def __init__(self, dropout_rate,filter_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(22, 25, kernel_size=filter_size, padding=get_same_padding(filter_size))
        self.pool1 = nn.MaxPool1d(kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(25)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(25, 50, kernel_size=filter_size, padding=get_same_padding(filter_size))
        self.pool2 = nn.MaxPool1d(kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(50)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv1d(50, 100, kernel_size=filter_size, padding=get_same_padding(filter_size))
        self.pool3 = nn.MaxPool1d(kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(100)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.conv4 = nn.Conv1d(100, 200, kernel_size=filter_size, padding=get_same_padding(filter_size))
        self.pool4 = nn.MaxPool1d(kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm1d(200)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(1400, 4)

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

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x