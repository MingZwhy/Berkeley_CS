import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################################
# TODO: Design your own neural network
# You can define utility functions/classes here
#######################################################################
pass
#######################################################################
# End of your code
#######################################################################


class conv_block2(nn.Module):
    def __init__(self, input_dim, output_dim, size=3, stride=1):
        super(conv_block2, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, size, stride, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_dim, output_dim, size, stride, padding='same')
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        
        return out
    
class conv_block3(nn.Module):
    def __init__(self, input_dim, output_dim, size=3, stride=1):
        super(conv_block3, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, size, stride, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_dim, output_dim, size, stride, padding='same')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(output_dim, output_dim, size, stride, padding='same')
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        
        return out

class MyNeuralNetwork(nn.Module):
        
    
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

        #######################################################################
        # TODO: Design your own neural network
        #######################################################################
        
        self.conv_block1 = conv_block2(3, 32)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block2 = conv_block2(32, 64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block3 = conv_block3(64, 128)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.relu1 = nn.ReLU()
        if p_dropout > 0.0:
            self.drop1 = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.relu2 = nn.ReLU()
        if p_dropout > 0.0:
            self.drop2 = nn.Dropout(p_dropout)
        self.fc3 = nn.Linear(in_features=1024, out_features=100)
        
        #######################################################################
        # End of your code
        #######################################################################
        
        '''
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=100)
        '''

    def forward(self, x):
        #######################################################################
        # TODO: Design your own neural network
        #######################################################################
        
        x = self.conv_block1(x)
        x = self.max_pool1(x)
        x = self.conv_block2(x)
        x = self.max_pool2(x)
        #x = self.conv_block3(x)
        #x = self.max_pool3(x)
        
        x = torch.flatten(x, 1)
        x = self.relu1(self.fc1(x))
        if self.p_dropout > 0.0:
            x = self.drop1(x)
        x = self.relu2(self.fc2(x))
        if self.p_dropout > 0.0:
            x = self.drop2(x)
        x = self.fc3(x)
        
        
        '''
        x = self.conv1(x)  # [bsz, 16, 32, 32]
        x = self.bn1(x)
        x = self.pool1(self.relu1(x))  # [bsz, 16, 16, 16]

        x = self.conv2(x)  # [bsz, 32, 16, 16]
        x = self.bn2(x)
        x = self.pool2(self.relu2(x))  # [bsz, 32, 8, 8]

        x = self.conv3(x)  # [bsz, 64, 4, 4]
        x = self.bn3(x)
        x = self.relu3(x)
        #x = self.pool3(self.relu3(x))
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        #print(x.shape)

        x = torch.flatten(x, 1)  # [bsz, 1024]
        x = self.relu5(self.fc1(x))  # [bsz, 256]
        x = self.relu6(self.fc2(x))
        x = self.fc3(x)  # [bsz, 100]
        '''
        
        return x
        #######################################################################
        # End of your code
        #######################################################################
