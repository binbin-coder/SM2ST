import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
# from .gatv2_conv_or import GATv2Conv as GATConv
from torch.nn.utils import spectral_norm

class encoding_mask_noise(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(encoding_mask_noise, self).__init__()   
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.enc_mask_token = nn.Parameter(torch.zeros(size=(1, in_dim)))
        self.reset_parameters_for_token()
        
    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token.data, gain=1.414)#
        
    def forward(self, x, mask_rate=0.5, replace_rate=0.05):
        # num_nodes = g.num_nodes()
        num_nodes = x.size()[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_token_rate = 1-replace_rate
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]
        
        if replace_rate > 0.0:
            num_noise_nodes = int(replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: -num_noise_nodes]]#int(mask_token_rate * num_mask_nodes)
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            # out_x[token_nodes] = torch.zeros_like(out_x[token_nodes])
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
            # out_x[noise_nodes] = torch.add(x[noise_to_be_chosen], out_x[noise_nodes]) 
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        # use_g = g.clone()
        return out_x, mask_nodes, keep_nodes

class random_remask(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(random_remask, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.dec_mask_token = nn.Parameter(torch.zeros(size=(1, out_dim)))
        self.reset_parameters_for_token()
        
    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.dec_mask_token.data, gain=1.414)
        
    def forward(self,rep,remask_rate=0.5):
        num_nodes = rep.size()[0]
        # num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        out_rep = rep.clone()
        out_rep[remask_nodes] = 0.0
        out_rep[remask_nodes] += self.dec_mask_token
        return out_rep, remask_nodes, rekeep_nodes


# class Encoder(nn.Module):
#     def __init__(self, mz_number, X_dim):
#         super(Encoder, self).__init__()
#         # self.encoding_mask_noise = encoding_mask_noise(hidden_dims)
#         # self.random_remask = random_remask(hidden_dims)
#         self.fc1 = nn.Linear(mz_number, 1024)
#         self.fc1_bn = nn.BatchNorm1d(1024)
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc2_bn = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, 64)
#         self.fc3_bn = nn.BatchNorm1d(64)
#         self.fc4 = nn.Linear(64, 8)
#         self.fc4_bn = nn.BatchNorm1d(8)
#         self.fc5 = nn.Linear(8, X_dim)
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
    
#     def forward(self, features, relu=False, mask = 0.0):
#         if mask:
#             mask_tensor = torch.bernoulli(torch.full_like(features, mask)).to(features.device)  # Random mask with 50% probability
#             features = features * mask_tensor  # Apply mask
#         h1 = F.relu(self.fc1_bn(self.fc1(features)))
#         h2 = F.relu(self.fc2_bn(self.fc2(h1)))
#         h3 = F.relu(self.fc3_bn(self.fc3(h2)))
#         h4 = F.relu(self.fc4_bn(self.fc4(h3)))
#         if relu:
#             return F.relu(self.fc5(h4))
#         else:
#             return self.fc5(h4)

class Encoder(nn.Module):
    def __init__(self, mz_number, X_dim, down_ratio):
        super(Encoder, self).__init__()
        self.dropout_rate = down_ratio
        
        self.fc1 = nn.Linear(mz_number, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        self.fc2 = nn.Linear(1024, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        self.fc3 = nn.Linear(256, 64)
        self.fc3_bn = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        self.fc4 = nn.Linear(64, 16)#8
        self.fc4_bn = nn.BatchNorm1d(16)#8
        self.dropout4 = nn.Dropout(self.dropout_rate)
        
        self.fc5 = nn.Linear(16, X_dim)
        
        # Initialize parameters
        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain('relu')
        # Initialize weights and biases for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use the Xavier initialization method to specify the gain value
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    # Initialize the bias to 0
                    nn.init.zeros_(module.bias)
    
    def forward(self, features, relu=False):
        # h1 = self.CustomDropout1(features)
        # h1 = F.relu(self.fc1_bn(self.fc1(h1)))
        h1 = F.relu(self.fc1_bn(self.fc1(features)))
        h1 = self.dropout1(h1)
        
        h2 = F.relu(self.fc2_bn(self.fc2(h1)))
        h2 = self.dropout2(h2)
        
        h3 = F.relu(self.fc3_bn(self.fc3(h2)))
        h3 = self.dropout3(h3)
        
        h4 = F.relu(self.fc4_bn(self.fc4(h3)))
        h4 = self.dropout4(h4)
        
        if relu:
            return F.relu(self.fc5(h4))
        else:
            return self.fc5(h4)


# class Decoder(nn.Module):
#     def __init__(self, mz_number, X_dim):
#         super(Decoder, self).__init__()
#         self.fc6 = nn.Linear(X_dim, 8)
#         self.fc6_bn = nn.BatchNorm1d(8)
#         self.fc7 = nn.Linear(8, 64)
#         self.fc7_bn = nn.BatchNorm1d(64)
#         self.fc8 = nn.Linear(64, 256)
#         self.fc8_bn = nn.BatchNorm1d(256)
#         self.fc9 = nn.Linear(256, 1024)
#         self.fc9_bn = nn.BatchNorm1d(1024)
#         self.fc10 = nn.Linear(1024, mz_number)
#         # Initialize parameters
#         self.init_weights()

#     def init_weights(self):
#         # Initialize weights and biases for all linear layers
#         gain = nn.init.calculate_gain('relu')
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 # Use the Xavier initialization method to specify the gain value
#                 nn.init.xavier_uniform_(module.weight, gain=gain)  
#                 if module.bias is not None:
#                     # Initialize the bias to 0
#                     nn.init.zeros_(module.bias)
    
#     def forward(self, z, relu=False):
#         h6 = F.relu(self.fc6_bn(self.fc6(z)))
#         h7 = F.relu(self.fc7_bn(self.fc7(h6)))
#         h8 = F.relu(self.fc8_bn(self.fc8(h7)))
#         h9 = F.relu(self.fc9_bn(self.fc9(h8)))
#         if relu:
#             return F.relu(self.fc10(h9))
#         else:
#             return self.fc10(h9)

class Decoder(nn.Module):
    def __init__(self, mz_number, X_dim, down_ratio):
        super(Decoder, self).__init__()
        self.dropout_rate = down_ratio
        
        self.fc6 = nn.Linear(X_dim, 16)#8
        self.fc6_bn = nn.BatchNorm1d(16)#8
        self.dropout6 = nn.Dropout(self.dropout_rate)
        
        self.fc7 = nn.Linear(16, 64)
        self.fc7_bn = nn.BatchNorm1d(64)
        self.dropout7 = nn.Dropout(self.dropout_rate)
        
        self.fc8 = nn.Linear(64, 256)
        self.fc8_bn = nn.BatchNorm1d(256)
        self.dropout8 = nn.Dropout(self.dropout_rate)
        
        self.fc9 = nn.Linear(256, 1024)
        self.fc9_bn = nn.BatchNorm1d(1024)
        self.dropout9 = nn.Dropout(self.dropout_rate)
        
        self.fc10 = nn.Linear(1024, mz_number)
        
        # Initialize parameters
        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain('relu')
        # Initialize weights and biases for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use the Xavier initialization method to specify the gain value
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    # Initialize the bias to 0
                    nn.init.zeros_(module.bias)
    
    def forward(self, z, relu=False):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h6 = self.dropout6(h6)
        
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h7 = self.dropout7(h7)
        
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        h8 = self.dropout8(h8)
        
        h9 = F.relu(self.fc9_bn(self.fc9(h8)))
        h9 = self.dropout9(h9)
        
        if relu:
            return F.relu(self.fc10(h9))
        else:
            return self.fc10(h9)

class Discriminator_A(torch.nn.Module):
    def __init__(self, X_dim):
        super(Discriminator_A, self).__init__()
        self.fc = torch.nn.Sequential(
            spectral_norm(nn.Linear(X_dim, 128)),# last best
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 8)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(8, 1)),
            nn.Sigmoid()
            # nn.Linear(X_dim, 64),
            # nn.LeakyReLU(0.2),
            # nn.Linear(64, 8),
            # nn.LeakyReLU(0.2),
            # nn.Linear(8, 1),
            # nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        # Initialize weights and biases for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use the Xavier initialization method to specify the gain value
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    # Initialize the bias to 0
                    nn.init.zeros_(module.bias)
    def forward(self, x):
        return self.fc(x)

class Discriminator_B(torch.nn.Module):
    def __init__(self, X_dim):
        super(Discriminator_B, self).__init__()
        self.fc = torch.nn.Sequential(
            nn.Linear(X_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            # nn.Linear(X_dim, 16),
            # nn.LeakyReLU(0.2),
            # nn.Linear(16, 4),
            # nn.LeakyReLU(0.2),
            # nn.Linear(4, 1),
            # nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        # Initialize weights and biases for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use the Xavier initialization method to specify the gain value
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    # Initialize the bias to 0
                    nn.init.zeros_(module.bias)
    def forward(self, x):
        return self.fc(x)