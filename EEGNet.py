import torch.nn as nn
import torch.nn.functional as F

# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self, nb_classes=2, Chans=2, Samples=128, dropoutRate=0.3, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        # block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.1, eps=1e-5)
        # Depthwise convolutional block
        self.depthwiseConv = nn.Conv2d(F1, F1*D, (Chans, 1), stride=1, groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D, momentum=0.1, eps=1e-5)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropoutRate)
        # block 2
        self.separableConv = nn.Conv2d(F1*D, F2, (1, 16), stride=1, padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2, momentum=0.1, eps=1e-5)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=dropoutRate)
                
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Dense output layer
        self.dense = nn.Linear(in_features=336, out_features=nb_classes, bias=True)
        
        # Activation function
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        # Depthwise convolutional block
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = F.elu(x,alpha = 2)
        x = self.avg_pool(x)
        x = self.dropout1(x)
        # Separable convolutional block
        x = self.separableConv(x)
        x = self.bn3(x)
        x = F.elu(x,alpha = 2)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        # Flatten layer
        x = self.flatten(x)
        # Dense output layer
        x = self.dense(x)
        #x = self.activation(x)
        return x
