import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.batchn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.batchn2 = nn.BatchNorm2d(64) 
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.batchn3 = nn.BatchNorm2d(128) 
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.batchn4 = nn.BatchNorm2d(256) 
        self.fc1 = nn.Linear(256*10*10, 1024)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 136)


        
    def forward(self, x):
        x = self.pool(F.relu(self.batchn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.batchn2(self.conv2(x))))
        
        x = self.pool(F.relu(self.batchn3(self.conv3(x))))
        
        x = self.pool(F.relu(self.batchn4(self.conv4(x))))
       

        x = x.view(x.size(0), -1)
        
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        
        return x