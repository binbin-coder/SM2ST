import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
# from .gatv2_conv_or import GATv2Conv as GATConv
from torch.nn.utils import spectral_norm

class Encoder(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(gene_number, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc2_bn = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 50)
        self.fc3_bn = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, 10)
        self.fc4_bn = nn.BatchNorm1d(10)
        self.fc5 = nn.Linear(10, X_dim)
    def forward(self, input, relu):
        h1 = F.relu(self.fc1_bn(self.fc1(input)))
        h2 = F.relu(self.fc2_bn(self.fc2(h1)))
        h3 = F.relu(self.fc3_bn(self.fc3(h2)))
        h4 = F.relu(self.fc4_bn(self.fc4(h3)))
        if relu:
            return F.relu(self.fc5(h4))
        else:
            return self.fc5(h4)


class Decoder(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Decoder, self).__init__()
        self.fc6 = nn.Linear(X_dim, 10)
        self.fc6_bn = nn.BatchNorm1d(10)
        self.fc7 = nn.Linear(10, 50)
        self.fc7_bn = nn.BatchNorm1d(50)
        self.fc8 = nn.Linear(50, 500)
        self.fc8_bn = nn.BatchNorm1d(500)
        self.fc9 = nn.Linear(500, 1000)
        self.fc9_bn = nn.BatchNorm1d(1000)
        self.fc10 = nn.Linear(1000, gene_number)
    def forward(self, z, relu):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        h9 = F.relu(self.fc9_bn(self.fc9(h8)))
        if relu:
            return F.relu(self.fc10(h9))
        else:
            return self.fc10(h9)
            
# class Encoder(nn.Module):
#     def __init__(self, mz_number, X_dim, down_ratio):
#         super(Encoder, self).__init__()
#         self.dropout_rate = down_ratio
        
#         self.fc1 = nn.Linear(mz_number, 1024)
#         self.fc1_bn = nn.BatchNorm1d(1024)
#         self.dropout1 = nn.Dropout(self.dropout_rate)
        
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc2_bn = nn.BatchNorm1d(256)
#         self.dropout2 = nn.Dropout(self.dropout_rate)
        
#         self.fc3 = nn.Linear(256, 64)
#         self.fc3_bn = nn.BatchNorm1d(64)
#         self.dropout3 = nn.Dropout(self.dropout_rate)
        
#         self.fc4 = nn.Linear(64, 16)#8
#         self.fc4_bn = nn.BatchNorm1d(16)#8
#         self.dropout4 = nn.Dropout(self.dropout_rate)
        
#         self.fc5 = nn.Linear(16, X_dim)
        
#         # Initialize parameters
#         self.init_weights()

#     def init_weights(self):
#         gain = nn.init.calculate_gain('relu')
#         # Initialize weights and biases for all linear layers
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 # Use the Xavier initialization method to specify the gain value
#                 nn.init.xavier_uniform_(module.weight, gain=gain)
#                 if module.bias is not None:
#                     # Initialize the bias to 0
#                     nn.init.zeros_(module.bias)
    
#     def forward(self, features, relu=False):
#         # h1 = self.CustomDropout1(features)
#         # h1 = F.relu(self.fc1_bn(self.fc1(h1)))
#         h1 = F.relu(self.fc1_bn(self.fc1(features)))
#         h1 = self.dropout1(h1)
        
#         h2 = F.relu(self.fc2_bn(self.fc2(h1)))
#         h2 = self.dropout2(h2)
        
#         h3 = F.relu(self.fc3_bn(self.fc3(h2)))
#         h3 = self.dropout3(h3)
        
#         h4 = F.relu(self.fc4_bn(self.fc4(h3)))
#         h4 = self.dropout4(h4)
        
#         if relu:
#             return F.relu(self.fc5(h4))
#         else:
#             return self.fc5(h4)


# class Decoder(nn.Module):
#     def __init__(self, mz_number, X_dim, down_ratio):
#         super(Decoder, self).__init__()
#         self.dropout_rate = down_ratio
        
#         self.fc6 = nn.Linear(X_dim, 16)#8
#         self.fc6_bn = nn.BatchNorm1d(16)#8
#         self.dropout6 = nn.Dropout(self.dropout_rate)
        
#         self.fc7 = nn.Linear(16, 64)
#         self.fc7_bn = nn.BatchNorm1d(64)
#         self.dropout7 = nn.Dropout(self.dropout_rate)
        
#         self.fc8 = nn.Linear(64, 256)
#         self.fc8_bn = nn.BatchNorm1d(256)
#         self.dropout8 = nn.Dropout(self.dropout_rate)
        
#         self.fc9 = nn.Linear(256, 1024)
#         self.fc9_bn = nn.BatchNorm1d(1024)
#         self.dropout9 = nn.Dropout(self.dropout_rate)
        
#         self.fc10 = nn.Linear(1024, mz_number)
        
#         # Initialize parameters
#         self.init_weights()

#     def init_weights(self):
#         gain = nn.init.calculate_gain('relu')
#         # Initialize weights and biases for all linear layers
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 # Use the Xavier initialization method to specify the gain value
#                 nn.init.xavier_uniform_(module.weight, gain=gain)
#                 if module.bias is not None:
#                     # Initialize the bias to 0
#                     nn.init.zeros_(module.bias)
    
#     def forward(self, z, relu=False):
#         h6 = F.relu(self.fc6_bn(self.fc6(z)))
#         h6 = self.dropout6(h6)
        
#         h7 = F.relu(self.fc7_bn(self.fc7(h6)))
#         h7 = self.dropout7(h7)
        
#         h8 = F.relu(self.fc8_bn(self.fc8(h7)))
#         h8 = self.dropout8(h8)
        
#         h9 = F.relu(self.fc9_bn(self.fc9(h8)))
#         h9 = self.dropout9(h9)
        
#         if relu:
#             return F.relu(self.fc10(h9))
#         else:
#             return self.fc10(h9)
