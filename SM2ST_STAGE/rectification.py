import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
import os


def standard(vec_1):
    return (vec_1-min(vec_1))/(max(vec_1)-min(vec_1))

def msi_sum(ms,threshold):
    ## x|y|ms_1|ms_2|ms_3|...;dataframe
    ms_n=pd.concat([ms.loc[:,["x","y"]],ms.iloc[:,2:].sum(axis=1)],axis=1)

    ms_n.columns=["x","y","sum"]

    ms_n.loc[:,"sum_standard"]=standard(ms_n.loc[:,"sum"])

    ms_sum= ms_n.pivot(index='x', columns='y', values="sum_standard")

    ms_sum=np.where(ms_sum<threshold,0,ms_sum)

    return ms_sum

def he_img(path):

    image = cv2.imread(path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    normalized_gray_image = 1-gray_image.astype(np.float32) / 255.0

    return normalized_gray_image

def padding(shape_large,shape_small):
    #dim_1_ratio
    dim_1_rate=shape_large[0]/shape_small[0]

    dim_2_rate=shape_large[1]/shape_small[1]
    
    #take max rate
    
    rate=int(max(dim_1_rate,dim_2_rate))+1

    #augment rate
    new_shape=shape_small*rate

    return new_shape,rate

def he_padding(ms_sum,HE_img,padding_value):
    # ms_sum type matrix np.array
    # HE_img type matrix np.array
    msi_tensor=torch.tensor(np.array(ms_sum),dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    HE_tensor=torch.tensor(np.array(HE_img),dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    size_he=np.array(HE_tensor.shape)[2:4]

    size_msi=np.array(msi_tensor.shape)[2:4]
    
    new_shape,rate=padding(size_he,size_msi)

    HE_tensor=F.pad(HE_tensor,
                pad=(0,new_shape[1]-size_he[1],
                    0,new_shape[0]-size_he[0]),value=padding_value)
    return HE_tensor, msi_tensor, rate
    
class spatial_transfor(Module):
    def __init__(self,device,grid_size):
        # grid_size= [padded_HE_height,padded_HE_width] list
        super(spatial_transfor,self).__init__()

        self.w=Parameter(torch.tensor(1.0,dtype=torch.float32,requires_grad=True,device=device))

        self.bias=Parameter(torch.tensor([[0],[0]],dtype=torch.float32,requires_grad=True,device=device))

        self.grid_size=[1,1]+grid_size

    def forward(self, x):
        self.diagonal_tensor = torch.diag(torch.stack([-self.w, self.w]))
        
        self.theta=torch.cat([self.diagonal_tensor,self.bias],dim=1)

        self.theta=self.theta.unsqueeze(0)
        
        grid = F.affine_grid(self.theta, size=torch.Size(self.grid_size), align_corners=False)

        warp_img = F.grid_sample(x, grid)

        return warp_img


def tensor_viewer(img):
    plt.imshow(img.squeeze(), cmap='gray')
    plt.colorbar()
    plt.title("Gray Scale Image")
    plt.show()

class fit():
    def __init__(self,num_epochs=500,lr=0.05,device=torch.device("cpu")):
        super(fit,self).__init__()
        self.num_epochs=num_epochs
        self.lr=lr
        self.device=device
        
    def train_AF(self,msi_tensor,he_tensor,grid_size):
        
        # msi_tensor tensor [1,1,h,w]
        # he_tensor tensor [1,1,h,w]
        # grid_size list [h,w]

        loss_func=nn.MSELoss()

        self.net=spatial_transfor(self.device,grid_size).to(self.device)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        # optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4) #, eps=1e-8, betas=(0.5, 0.999)

        msi_tensor=msi_tensor.to(self.device)
        he_tensor=he_tensor.to(self.device)
        
        for epoch in tqdm(range(self.num_epochs)):
            self.net.train()
            optimizer.zero_grad()
            Y_hat= self.net(msi_tensor)
            # print(Y_hat.shape,he_tensor.shape)
            l = loss_func(Y_hat,he_tensor)
            l.backward()
            optimizer.step()
            # print(f"{epoch}:{l}")

def torch_af_transfor(af_theta,shape,org_shape,points):

    W,H=org_shape

    points_norm = torch.stack([
        (points[:, 0] / W)*2-1 ,
        (points[:, 1] / H)*2-1], dim=0).T

    points_norm[:,0]= (points_norm[:,0]- af_theta[0,2])/af_theta[0,0]

    points_norm[:,1]= (points_norm[:,1]- af_theta[1,2])/af_theta[1,1]

    
    W_prime, H_prime = shape

    transformed_points = torch.stack([
        (points_norm[:, 0]+1)/2 * W_prime,
        (points_norm[:, 1]+1)/2 * H_prime
    ], dim=0)

    return transformed_points.T

def torch_af_transfor(af_theta,shape,org_shape,points):

    H, W=org_shape

    points_norm = torch.stack([
        (points[:, 0] / W)*2-1,
        (points[:, 1] / H)*2-1], dim=0).T
    #Complete af
    line3=torch.tensor([[0,0,1]],dtype=torch.float32)
    al_3=torch.cat([af_theta,line3],dim=0)
    af_norm=torch.inverse(al_3)
    #Complete point
    ones=torch.ones(points_norm.shape[0],1)
    points_norm=torch.cat([points_norm,ones],dim=1)
    points_norm=points_norm.T
    # print(points_norm.shape)
    # print(af_norm.shape)
    points_norm=torch.mm(af_norm,points_norm).T

    # print(points_norm)
    H_prime, W_prime = shape

    transformed_points = torch.stack([
        (points_norm[:, 0]+1)/2 * W_prime,
        (points_norm[:, 1]+1)/2 * H_prime
    ], dim=0)

    return transformed_points.T

def fix_seed(seed):
    #seed = 2025
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'