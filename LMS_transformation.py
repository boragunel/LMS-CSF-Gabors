

### The code stated below is adapted from the previous VGG16 model and it refers to the RGB/LMS transformations:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchsummary import summary
import numpy as np
import math
import os
import psutil
import time


def opponency_encoder(self,data_tensor):                    
    num, channel, row, col=data_tensor.shape                  
    inputshape=(num,int(channel/3),row,col)                   
    r = data_tensor[:, 0, :, :]                               
    g = data_tensor[:, 1, :, :]
    b = data_tensor[:, 2, :, :]

    gamma_value=2.2                                          
    r= r.pow(1/gamma_value)                                  
    g = g.pow(1/gamma_value)
    b = b.pow(1/gamma_value)

    '''
    RGB_tensor = torch.stack([r, g, b])
    gamma_corrrected_RGB = torchvision.transforms.functional.adjust_gamma(RGB_tensor, gamma=2.2)
    r = gamma_corrrected_RGB[0]
    g = gamma_corrrected_RGB[1]
    b = gamma_corrrected_RGB[2]
    '''
    
    
    
    x= 0.412453 * r + 0.357580 * g + 0.180423 * b         
    y= 0.212671 * r + 0.715160 * g + 0.072169 * b         
    z= 0.019334 * r + 0.119193 * g + 0.950227 * b
    
    '''
    LMS_trans_matrix=torch.tensor([                        ###Desbribes the transformation matrix in order to convert from XYZ colour space to LMS
    [0.3897, 0.6890, -0.0787],                             ###Matrix obtained from Hunter-Pointer-Estevez approximation
    [-0.2298, 1.1834, 0.0464],
    [0.0000, 0.0000, 1.0000]
    ])

    col_space_matrix=torch.stack([x,y,z], dim=0)
    col_space_matrix=col_space_matrix.to(device)
    LMS_trans_matrix=LMS_trans_matrix.to(device)

    l=(torch.matmul(col_space_matrix, LMS_trans_matrix.T))[0]
    m=(torch.matmul(col_space_matrix, LMS_trans_matrix.T))[1]
    s=(torch.matmul(col_space_matrix, LMS_trans_matrix.T))[2]     ###Performs matrix multiplication between the tensors and final step for converting RGB to LMS matrix
    '''

    l=0.3897*x + 0.6890*y + -0.0787*z
    m=-0.2298*x + 1.1834*y + 0.0464*z
    s=0.0000*x + 0.0000*y + 1.0000*z


    I = ((l + m + s) / 3).reshape(inputshape)                
    II = (1 - ((l + m + s ) / 3)).reshape(inputshape)
    R = torch.clamp(l / (l + m),min=0.0)                  
    G = torch.clamp(1 - (l / (l + m )),min=0.0)
    B = torch.clamp(s / (l + m),min=0.0)
    Y = torch.clamp(1- (s / (l + m)),min=0.0)
    RG = torch.clamp(R ,min=0.0).reshape(inputshape)
    GR = torch.clamp(G ,min=0.0).reshape(inputshape)
    BY = torch.clamp(B ,min=0.0).reshape(inputshape)
    YB = torch.clamp(Y ,min=0.0).reshape(inputshape)
    return torch.cat((I,II,RG,GR,BY,YB), 1)
 

### The function below refer to the opponency encoder from the reference vgg16 model

def opponency_encoder(self,data_tensor):
    num, channel, row, col=data_tensor.shape
    inputshape=(num,int(channel/3),row,col)
    r = data_tensor[:, 0, :, :]
    g = data_tensor[:, 1, :, :]
    b = data_tensor[:, 2, :, :]
    I = ((r + g + b) / 3).reshape(inputshape)
    R = torch.clamp(r - (g + b) / 2,min=0.0)
    G = torch.clamp(g - (r - b) / 2,min=0.0)
    B = torch.clamp(b - (r + g) / 2,min=0.0)
    Y = torch.clamp((r + g) / 2 - (r - g) / 2 - b,min=0.0)
    RG = torch.clamp(R - G,min=0.0).reshape(inputshape)
    GR = torch.clamp(G - R,min=0.0).reshape(inputshape)
    BY = torch.clamp(B - Y,min=0.0).reshape(inputshape)
    YB = torch.clamp(Y - B,min=0.0).reshape(inputshape)
    return torch.cat((I,RG,GR,BY,YB),1)
    

    #return torch.cat((I,II,RG,GR,BY,YB),1)                       

