    #INTRODUCTION TO APPLICATIONS AND COMPUTER SYSTEMS NECESSARY TO RUN THE PROGRAMS:


#REMEMBER: The below codes, commands and methods refer to the latest versions of windows.
#All of the above might not be compatible for mac users

#Install python along with an IDE (preferrably VS-code) for starters
#You will also need to have an installing package:
#For downloading PIP3:

#1. Go to https://bootstrap.pypa.io/pip/pip.py and download the python script (you can do this by going to 'More Tools, Save Page, and finally choose the directory and save as a python script)
#2. open the command prompt by typing 'cmd' into the start tab
#3. into the command prompt, type 'cd <directory where the downloaded script is located>' and press enter
#4. Once the directory has been set up, type 'py get-pip.py'into the command prompt and wait for package to be downloaded

#for checking that the file has been downloaded, you can type the command 'pip --version' into the cmd prompt
#if you want to uninstall pip from the directory, you can use 'python -m pip uninstall pip'


#Once has been installed, you can use it to download 'Torch'
#The Easisest way to do this is by going to 'https://pytorch.org/' and choosing the most suitable command for your computer
#The command used to run the studies were: 'pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117'
#to check if Torch is installed, you can use the following code:

import torch
print(torch.__version__)

#IMPORTANT: In order to run the specific programs, a suitable version of cuda should be installed on your computer
#In order to check if this has already been installed in your computer, you type 'nvcc --version' into the cmd prompt
#If it is already present, it will print out the version number. However of it doesn't come up with a results,
#it will need to be downloaded manually
#Go to 'https://developer.nvidia.com/cuda-toolkit-archive' and select an appropriate version of CUDA for your computer (for the specific study we used 'CUDA Toolkit 11.7.0 (May 2022), Versioned Online Documentation')
#Choose windows, and the version of windows you are using (10 & 11), then for select installer type, select 'exe local'
#Once appropriate options have been picked, you can press 'download' and wait for the installation
#Once the version has been downloaded, you can check using the command prompt and move on to the next stage

#The next stage is importing other packages
#on your python script, type the lines:

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
from torchvision.models import ResNet

#If certain packages have not been installed, the editor will underline the package in red
#The packages wil also need to be downloaded manually, the specific commands can be found using the web browser
#Once finalised, the models can be run.








