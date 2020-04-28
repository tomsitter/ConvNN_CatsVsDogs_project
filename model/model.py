import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch

class CatsVsDogsModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)
               
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 =  nn.Linear(512, num_classes)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        
        # in order to determine # of features to pass to Linear layer
        # we have to get the total size of the last output layer.
        # We will flatten it and get the size, assignings to an internal variable to use when
        # defining input for the first linear layer
        x = torch.flatten(x, 1, -1)
        if self._to_linear is None: 
            self._to_linear = x.shape[1]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
