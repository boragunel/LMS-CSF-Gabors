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
print(torch.__version__)
print(torch.cuda.get_device_name(0))
import time
from torchvision.models import ResNet
#import shutil
#above represent the imports


Batch_size=128  #128 before, this sets the number of samples that will be picked out from the dataset for training or testing
transform = transforms.Compose([
#     transforms.CenterCrop(224),
    transforms.Resize(32),                 #If the image has size larger than 32x32, the code will reshape it into 32x32. If it's smaller than 32x32, it will reamined unchanged
    transforms.RandomCrop(32,padding=4),   #Random crop is performed on image and padding adds 0 weights around the image
    transforms.RandomHorizontalFlip(),     #Performs random horizontal flip to the image with a chance of 50% chance, which common for increase in diversity for training data
    transforms.ToTensor(),                 #Converts the image from PIL format to PyTorch tensor values
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    #Nomralizes pixel values from the image. It suybstracts 0.5 from each channel (R, G, B) and then divides them by 0.5
])

a="C:\\Users\\bmg24\\OneDrive - University of Sussex\\Desktop\\Python_files\\ecoset-10\\test"    #Specify train directory
b="C:\\Users\\bmg24\\OneDrive - University of Sussex\\Desktop\\Python_files\\ecoset-10\\train"     #Specify test directory

trainingset1=datasets.ImageFolder(root=a,transform=transform)    #'datasets.ImageFolder' purpose is to create a directory where each subdirectory is a class and images within belong to that class
testset1=datasets.ImageFolder(root=b,transform=transform)        #the specified 'root' refers to the required directory pathway, and 'transform' refers to the transomfration techniques (which was defined previously)
'''trainingset1=datasets.ImageFolder(root="../data/Cifar10/train_cifar10",transform=transform)
testset1=datasets.ImageFolder(root="../data/Cifar10/test_cifar10",transform=transform)'''
trainloader1=DataLoader(trainingset1,batch_size=Batch_size,shuffle=True,num_workers=4) #variable for loading data
testloader1=DataLoader(testset1,batch_size=Batch_size,shuffle=False,num_workers=4)     #shuffle and batch size are obvious, 'num_workers' when higher than 0 it loads data using multiple threads

Batch_Size = 256   
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}  #Sets dictionary which index VGG type along with its model arcihtecture
   #In this case we will be using VGG16 which is composed of 13 convolutional layers and 5 max pooling layers
   #The model architecture: first 2 layer set with 64 kernels, M layer, second 2 layer set with 128 kernels, 
   #third 3 layer set with 256 kernels, fourth 3 layer set with 512 kernels, and the last 3 layer set with 512 kernels


'''cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}'''

class GaborConv2d(nn.Module):
    def __init__(
        self,
        in_channels,                   #input channels refers to input channel numbers such as (3,32,32)
        out_channels,                  #output channels refers to the chanel numbers of the final product from after the kernel operation
        kernel_size,                   #size/dimensions of the filter
        stride=1,                      #the number of steps the kernel moves
        padding=0,                     #number of 0 weighted values added to the sides of the image
        dilation=1,                    #dilation value is the strretching proportion of the kernel. a dilation of 1 would mean no 0 weight values would be found between kernel values
        groups=1,                      #defines the dicision between the connectivity between the layers. 
        bias=False,                    #Bias refers to bias for certain features
        padding_mode="zeros",          #padding mode refers to the weight value the paddings will have
    ):
        super().__init__()

        self.is_calculated = False   #Will be referred later

        self.conv_layer = nn.modules.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        ) #Note that the above are the same as first values however, they have been pulled from nn.module superclass
        self.kernel_size = self.conv_layer.kernel_size   #

        # small addition to avoid division by zero
        self.delta = 1e-3      #defines delta as 0.001

        # freq, theta, sigma are set up according to S. Meshgini,
        # A. Aghagolzadeh and H. Seyedarabi, "Face recognition using
        # Gabor filter bank, kernel principal component analysis
        # and support vector machine"
        self.freq = nn.Parameter((math.pi / 2) * math.sqrt(2) ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),requires_grad=True, ) #calculate frequency for gabor filter
        self.theta = nn.Parameter((math.pi / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),requires_grad=True,) #calculates theta
        self.sigma = nn.Parameter(math.pi / self.freq, requires_grad=True) #calculates sigma value
        self.psi = nn.Parameter(math.pi * torch.rand(out_channels, in_channels), requires_grad=True) #calculates psi value
        self.x0 = nn.Parameter(torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False)  #calculates x value
        self.y0 = nn.Parameter(torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False)  #calculates y value
        self.y, self.x = torch.meshgrid(                                                #creates a grids. self.y has same values horizontally and self.x has same values vertically
            [torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),            #creates equally spaced values between -self.x0+1 and self.x0+0 where the space size is set to self.kernel_size[0]
             torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),])          ##creates equally spaced values between -self.y0+1 and self.y0+0 where the space size is set to self.kernel_size[1]
        self.y = nn.Parameter(self.y)                   #This operation converts self.y and self.x into a learnable parameter
        self.x = nn.Parameter(self.x)
        self.weight = nn.Parameter(torch.empty(self.conv_layer.weight.shape, requires_grad=True),requires_grad=True,)   #changes the weight to learnable parameters with empty tensor values


        self.register_parameter("freq", self.freq)            #self.register_parameter is a built-in pytorch method for adding custom tensors as parameters
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):    #forward function ensures that the weights of the Gabor function are properly calculated before applying it to image data
        if self.training:
            self.calculate_weights()    #checks module is currently in training, if it is it activates the custom calculate weights function (which is defined below)
            print(self.conv_layer.weight.data.shape)
            self.is_calculated = False  
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()    #checks that training is not in process. If self.is calculated remains false, it calculates the weights using the custom function and changes it to true
                self.is_calculated = True
        return self.conv_layer(input_tensor)   #preforms the forward pass by applying convolutional operation to the input tensor using gabor filter weights
                                               #The output of this convolutional layer is returned as the final output of the GaborConv2D module

    def calculate_weights(self):                                    #the function as it says is responsible for calculating weight values of the gabor filter kernels
        for i in range(self.conv_layer.out_channels):               #For each pair of output and input, a seperate gabor filter is generated     
            for j in range(self.conv_layer.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)          #the repeated lines expand the parameters to have the same size as self.y
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)  #calculates the rotated x coordinates based on angle theta
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta) #calculates the rotated y coordinates based on angle theta

                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2))   #calculates the Gaussian
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)           #the above g operations is for calculating the gabor
                self.conv_layer.weight.data[i, j] = g        #finally the calculated gabor filter g is assigned to the weight tensor of the corresponding fillter in the convolutional layer
#These features will be learned and adapted during training

class GaborConvFixed(nn.Module):     #Note that this calss is diffferent in the sense that, the Gabor filter values and weights are fixed and are not changed during training unless acted by an external source
    def __init__(self,
          in_channels,
          out_channels,
          input_dict,
          kernel_size,
          stride=1,
          padding=0,
          dilation=1,
          groups=1,
          bias=False,
          padding_mode="zeros",
          ):
        super().__init__()
        self.input_dict=input_dict #defines input_dict dictionary as a variable using self.
        if self.input_dict==None:
            self.input_dict = {  # 'ksize': (127, 127),
            'ksize': (31, 31),  #sets kernel size as (31,31)
            'gammas': [0.5],
        #           'bs': np.linspace(0.4, 2.6, num=3),  # 0.4, 1, 1.8, 2.6
        #           'bs': np.linspace(0.4, 2.6, num=5),
        'bs': np.linspace(1, 2.6, num=3).tolist(),   #bs is set to be 3 equally spaced values between and including 1 and 2.6
        #           'bs': np.linspace(1, 2.6, num=5),
        #           'sigmas': [4, 8, 16],  # , 32
        'sigmas': [8],
        'thetas': np.linspace(0, np.pi, 4, endpoint=False).tolist(),   #theta is set to be equal to 4 equally spaced numbers between 0 and pi (without including total pi) and converted into a python list   
        'psis': [np.pi / 2, 3 * np.pi / 2]}                            
    
        self.ksize = self.input_dict["ksize"]     #At this step, the variables in a gabor filter equation are set to the corresponding items within the 'input_dict' 
        self.sigmas = self.input_dict["sigmas"]
        self.bs = self.input_dict["bs"]
        self.gammas = self.input_dict["gammas"]
        self.thetas = self.input_dict["thetas"]
        self.psis = self.input_dict["psis"]

        self.conv_layer = nn.modules.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
            )
      
        self.weight = nn.Parameter(torch.empty(self.conv_layer.weight.shape, requires_grad=False),requires_grad=False,)    #the weights of the convolutional layer are initialized during this step

    def calc_lambda(self, sigma, bandwidth):            #gives the equation for calculating lambda
      p = 2**bandwidth                             
      c = np.sqrt(np.log(2)/2)
      return sigma * np.pi / c  * (p - 1) / (p + 1)
    
    def forward(self, input_tensor):                    #calculates the forward pass  
      self.calculate_weights()
      return self.conv_layer(input_tensor) #it applies the calculated weights to the 'input_tensor' 
    
    def calculate_weights(self):       #the function calculates the weights through iterating on every value of every parameter find within the inputc_dict
      for i in range(self.conv_layer.out_channels):
          for j in range(self.conv_layer.in_channels):
              for sigma in self.sigmas:
                for theta in self.thetas:
                    # for lambd in lambdas:
                    for b in self.bs:
                        lambd = self.calc_lambda(sigma, b)
                        for gamma in self.gammas:
                            for psi in self.psis:
                                gf = cv2.getGaborKernel(self.ksize, sigma, theta,lambd, gamma, psi, ktype=cv2.CV_64F)   #Based on the values, thorugh every iteration it produces a Gabor filter with specified weights
                                self.conv_layer.weight.data[i, j] = torch.tensor(gf)                                    #this code transofrms the value into a torch tensor and assignes it as weight values for the self.convolutional.layer
    
    

class VGG(nn.Module):      #This class defines the VGG network, however it is customly altered
    
    def __init__(self, vgg_name, param=None):
        super(VGG, self).__init__()
        self.param=param              #optional parameter that can be used to specify Gabor parameters
        self.Gabor_out_channels=64    #the 64 represents the number of Gabor output channels for the GaborConv2D
        '''if param:
          self.g0=GaborConvFixed(in_channels=3, out_channels= 24, input_dict=self.param,kernel_size=(31, 31) ,padding=15)
        else:'''
        self.g0=GaborConv2d(in_channels=6, out_channels= self.Gabor_out_channels, kernel_size=(31, 31),padding=15) #sets input channels are set to 5, the output channels are set to 64 ### changed input channels from 5 to 6 as a result of adding the inverse luminance channel to the opponency encoder
        
        self.features = self._make_layers(cfg[vgg_name])     #make_layers() function is called and you can specify the VGG model name within
        self.classifier = nn.Sequential(                     
            nn.Linear(512,512),          #takes input 512 and produces output 512. this suggests it has 512 input and 512 output neurons
            nn.ReLU(True),               
            nn.Dropout(0.2),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(0.2),              #Dropout sets a fraction of the activations to 0 during training to prevent overfitting
            nn.Linear(512,10),
        )
#         self.classifier = nn.Linear(512,10)

        self._initialize_weight()         #Creates an instance of initialize weights function defined below
        
    def forward(self, x):             #function performs the forward pass
      x=self.opponency_encoder(x)     #creates x variable to give name to opponency_encoder method defined below
      out = self.features(self.g0(x)) #creates variable out to define application of self.features function on self.g0 
      # 在进入
      out = out.view(out.size(0), -1)
      #out = self.classifier(out)
      return out
    
    
    def opponency_encoder(self,data_tensor):                    #defines function for colour opponency encoding
        num, channel, row, col=data_tensor.shape                  #data possess shape batch_size, channel, height, width
        inputshape=(num,int(channel/3),row,col)                   #int(channel/3) divides the channel dimension into three
        r = data_tensor[:, 0, :, :]                               #As channel was the second dimension of data_tensor, for each colour a different channel within is specified
        g = data_tensor[:, 1, :, :]
        b = data_tensor[:, 2, :, :]

        gamma_value=2.2                                          ### performs gamma correction
        r= r.pow(1/gamma_value)                                  ### the gamma values is set to a standard of 2.2
        g = g.pow(1/gamma_value)
        b = b.pow(1/gamma_value)
    
        '''
        RGB_tensor = torch.stack([r, g, b])
        gamma_corrrected_RGB = torchvision.transforms.functional.adjust_gamma(RGB_tensor, gamma=2.2)
        r = gamma_corrrected_RGB[0]
        g = gamma_corrrected_RGB[1]
        b = gamma_corrrected_RGB[2]
        '''
        
        
        
        x= 0.412453 * r + 0.357580 * g + 0.180423 * b          ###Conversion from RGB to XYZ Colour Space
        y= 0.212671 * r + 0.715160 * g + 0.072169 * b          ###Number and code Extracted from Kornia AI library
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
    
    
        I = ((l + m + s) / 3).reshape(inputshape)                 #Represents illumination value. Uses reshape value to convert it into desired tensor shape
        II = (1 - ((l + m + s ) / 3)).reshape(inputshape)
        R = torch.clamp(l / (l + m),min=0.0)                  
        G = torch.clamp(1 - (l / (l + m )),min=0.0)
        B = torch.clamp(s / (l + m),min=0.0)
        Y = torch.clamp(1- (s / (l + m)),min=0.0)
        RG = torch.clamp(R ,min=0.0).reshape(inputshape)
        GR = torch.clamp(G ,min=0.0).reshape(inputshape)
        BY = torch.clamp(B ,min=0.0).reshape(inputshape)
        YB = torch.clamp(Y ,min=0.0).reshape(inputshape)       #Process calculates R, B, G, Y values according to real life inspiration LGN receptive fields
        return torch.cat((I,II, RG, GR, BY, YB), 1)
    
    '''
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
        

        #return torch.cat((I,II,RG,GR,BY,YB),1)                       #Concatenates the values along dimension indexed as 1
    '''

        # make layers

    def _make_layers(self, cfg):                #The function builds the desired archistecture of the CNN
        layers = []                             #creates an empty list called layers
        in_channels = self.Gabor_out_channels   #sets in_channel variable the same self.Gabor_out_channels which is specified to be 64
        for x in cfg:                           
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)] #adds a max poolong layer with a kernel size of 2 and stride value of 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), # adds a convolutional layer of kernel size 3 and a padding value of 1
                           nn.BatchNorm2d(x),                                   #brings batch normalization function
                           nn.ReLU(inplace=True)]  # RelU
                in_channels = x                                                 #for the end of the function it brings x as the in_channel value
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)                                           #passes the created layers inyo nn.Sequential
    
    # 初始化参数
    def _initialize_weight(self):                                              #function for initialising weights and biases for the layers of the CNN network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier is used in VGG's paper
                nn.init.xavier_normal_(m.weight.data)                          #if object type is of 'convolutional layer', performs xaviers normalization
                if m.bias is not None:                                         #if biases are present, this line of code initializes it to zero
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):                                #if object type is of 'batch normalization', sets the weights to a value of 1. the bias is also set to 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):                                     #if the object is of type 'fully connected layer, 
                m.weight.data.normal_(0, 0.01)                                 #initializes weights to 0 with sd of 0.01
                m.bias.data.zero_()                                            #also sets the bias value to zero
    input_dict = {  # 'ksize': (127, 127),
            'ksize': (31, 31),
            'gammas': [0.5],
            #           'bs': np.linspace(0.4, 2.6, num=3),  # 0.4, 1, 1.8, 2.6
            #           'bs': np.linspace(0.4, 2.6, num=5),
            'bs': np.linspace(1, 2.6, num=3).tolist(),
            #           'bs': np.linspace(1, 2.6, num=5),
            #           'sigmas': [4, 8, 16],  # , 32
            'sigmas': [8],
            'thetas': np.linspace(0, np.pi, 4, endpoint=False).tolist(),
            'psis': [np.pi / 2, 3 * np.pi / 2]}
    
      
device = 'cuda' if torch.cuda.is_available() else 'cpu'                                     #cheks and sets the device for torch.cuda, if it is not avialble, it sets the device on cpu
net = VGG('VGG16').to(device)                                                               #creates an instance where the VGG16 neural network model is loaded to the device
if device == 'cuda':
    net = nn.DataParallel(net)                                                              #if device is run on GPU, this allows youto parallelize the computations on multiple GPU's in the computer to fasten the training etc.
    # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
    torch.backends.cudnn.benchmark = True                                                   #Calls algorithms from cuda DNN library to improve process, however algorithms and model architecture must remain constant thorugh the training

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)    #Loads a data with a single image sample through random chance, utilizing multiple threads
    mean = torch.zeros(3)                                                                           #creates a single dimension torch tensor with zero values
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()                                                       #calculates the mean for all the values in that specific dimension
            std[i] += inputs[:,i,:,:].std()                                                         #does the same and extracts the standard deviation
    mean.div_(len(dataset))                                                                         #divides the tjhe found mean by number of samples in th dataset
    std.div_(len(dataset))
    return mean, std                                                                                #returns the final mean and standard deviation


def get_acc(outputs, label):                                                                        #Calculates the accuracy of the process
    total = outputs.shape[0]                                                                        #takes size of the first dimension of the outputs (batch size, number of classes) which will give the batch size
    probs, pred_y = outputs.data.max(dim=1)                                                         
    correct = (pred_y == label).sum().data
    return correct / total


class EarlyStopping:                                                                               
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience                                                                      #Defines patience for the class
        self.verbose = verbose                                                                        #sets self.verbose as false
        self.counter = 0                                                                              
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf                                                                    #defines validation loss time as infinite
        self.delta = delta

    def __call__(self, val_loss, model):                                                             #sets behaviour of a class when an instance is called as a function
        score = -val_loss                                                                             
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0                                                                        #The purpose of this function is to stop the training completely if the training doesn't improve after 7 trials

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss                                                              

def train(net, trainloader, testloader, epochs, optimizer , criterion, scheduler , path = './model.pth', writer = None ,verbose = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                                     
    best_acc = 0
    train_acc_list, test_acc_list = [],[]                                                       
    train_loss_list, test_loss_list = [],[]                                                     
    lr_list  = []                                                                               
    for i in range(epochs):
        start = time.time()                                                                     
        train_loss = 0                                                                          
        train_acc = 0                                                                           
        test_loss = 0                                                                           
        test_acc = 0                                                                            
        if torch.cuda.is_available():
            net = net.to(device)                                                                
        net.train()
        for step,data in enumerate(trainloader,start=0):                                        
            im,label = data                                                                     
            im = im.to(device)                                                                  
            label = label.to(device)                                                            

            optimizer.zero_grad()                                                               
            # 释放内存
            if hasattr(torch.cuda, 'empty_cache'):                                              
                torch.cuda.empty_cache()                                                        
            # formard
            outputs = net(im)                                                                   
            loss = criterion(outputs,label)                                                     
            # backward
            loss.backward()                                                                     
            # 更新参数
            optimizer.step()                                                                    

            train_loss += loss.data
            # probs, pred_y = outputs.data.max(dim=1) # 得到概率
            # # 正确的个数
            # train_acc += (pred_y==label).sum().item()
            # # 总数
            # total += label.size(0)
            train_acc += get_acc(outputs,label)
            # 打印下载进度
            rate = (step + 1) / len(trainloader)
            a = "*" * int(rate * 50)
            b = "." * (50 - int(rate * 50))
            print('\r train {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epochs,int(rate*100),a,b),end='')
        train_loss = train_loss / len(trainloader)
        train_acc = train_acc * 100 / len(trainloader)
        if verbose:
            train_acc_list.append(train_acc.item())
            train_loss_list.append(train_loss.item())
    #     print('train_loss:{:.6f} train_acc:{:3.2f}%' .format(train_loss ,train_acc),end=' ')  
        # 记录学习率
        lr = optimizer.param_groups[0]['lr']
        if verbose:
            lr_list.append(lr)
        # 更新学习率
        scheduler.step(train_loss)
        if testloader is not None:
            net.eval()
            with torch.no_grad():
                for step,data in enumerate(testloader,start=0):
                    im,label = data
                    im = im.to(device)
                    label = label.to(device)
                    # 释放内存
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    outputs = net(im)
                    loss = criterion(outputs,label)
                    test_loss += loss.data
                    # probs, pred_y = outputs.data.max(dim=1) # 得到概率
                    # test_acc += (pred_y==label).sum().item()
                    # total += label.size(0)
                    test_acc += get_acc(outputs,label)
                    rate = (step + 1) / len(testloader)
                    a = "*" * int(rate * 50)
                    b = "." * (50 - int(rate * 50))
                    print('\r test  {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epochs,int(rate*100),a,b),end='')
            test_loss = test_loss / len(testloader)
            test_acc = test_acc * 100 / len(testloader)
            if verbose:
                test_loss_list.append(test_loss.item())
                test_acc_list.append(test_acc.item())
            end = time.time()
            print(
                '\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                    i + 1, epochs, train_loss, train_acc, test_loss, test_acc,lr), end='')
        else:
            end = time.time()
            print('\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(i+1,epochs,train_loss,train_acc,lr),end = '')
        time_ = int(end - start)
        h = time_ / 3600
        m = time_ % 3600 /60
        s = time_ % 60
        time_str = "\tTime %02d:%02d" % ( m, s)
        # ====================== 使用 tensorboard ==================
        if writer is not None:
            writer.add_scalars('Loss', {'train': train_loss,
                                    'valid': test_loss}, i+1)
            writer.add_scalars('Acc', {'train': train_acc ,
                                   'valid': test_acc}, i+1)
            writer.add_scalars('Learning Rate',lr,i+1)
        # =========================================================
        # 打印所用时间
        print(time_str)
        # 如果取得更好的准确率，就保存模型
        if test_acc > best_acc:
            torch.save(net,path)
            best_acc = test_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['test_acc'] = test_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['test_loss'] = test_loss_list
    Lr = lr_list
    return Acc, Loss, Lr


import matplotlib.pyplot as plt





def main():
    import torch.optim as optim
    optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,verbose=True,patience = 5,min_lr = 1e-100) # 动态更新学习率
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)

    import time
    epochs = 1
    import os
    if not os.path.exists('./model'):
        os.makedirs('./model')
    else:
        print('文件已存在')
    save_path = './model/OpponencyEcoGaborVGG16.pth'
    Acc, Loss, Lr = train(net, trainloader1, testloader1, epochs, optimizer, criterion, scheduler, save_path, verbose = True)

    def plot_history(epochs, Acc, Loss, Lr):
        plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        epoch_list = range(1,epochs + 1)
        plt.plot(epoch_list, Loss['train_loss'])
        plt.plot(epoch_list, Loss['test_loss'])
        plt.xlabel('epoch')
        plt.ylabel('Loss Value')
        plt.legend(['train', 'test'], loc='upper left')
        fig1=plt.gcf()
        fig1.savefig(os.path.join(os.getcwd(), "EcoVGG_loss_v_time"))
        
        plt.plot(epoch_list, Acc['train_acc'])
        plt.plot(epoch_list, Acc['test_acc'])
        plt.xlabel('epoch')
        plt.ylabel('Acc Value')
        plt.legend(['train', 'test'], loc='upper left')
        fig2=plt.gcf()
        fig2.savefig(os.path.join(os.getcwd(), "EcoVGG_accuracy_v_time"))
        

        plt.plot(epoch_list, Lr)
        plt.xlabel('epoch')
        plt.ylabel('Train LR')
        fig3=plt.gcf()
        fig3.savefig(os.path.join(os.getcwd(), "EcoVGG_Logistics_regression_v_time"))

        #figures={"loss_v_time":1, "accuracy_v_time":2, "Logistics_regression_v_time":3}
        #for f in figures.items():
        #    shutil.copytree(os.path.join(os.getcwd(),f),
        #         os.path.join("C:\\Users\\bmg24\\Desktop\\Python_files\\Learning_curve_images\\", f))

    plot_history(epochs, Acc, Loss, Lr)



if __name__ == "__main__":
    main()


