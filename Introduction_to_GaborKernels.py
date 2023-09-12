    
    
    
    ### This script rfeers to an introfduction to the mapping and visualization of Gabor kernels


import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.io import read_image    
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os



   ### The below code is for visualization with reference to slight Gabor parameter changes

ksize=30
sigma=3
theta=np.pi/4
lamda=np.pi/4
gamma=0.5
phi=np.pi
#kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
#plt.imshow(kernel, cmap='rainbow')
#fig1=plt.gcf()
#fig1.savefig(os.path.join(os.getcwd(), 'images_for_python/gabor_initial-gamma'))

kernel1= cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
kernel2 = cv2.getGaborKernel((ksize,ksize), sigma+3, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
kernel3 = cv2.getGaborKernel((ksize,ksize), sigma, theta*2, lamda, gamma, phi, ktype=cv2.CV_32F)
kernel4 = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda*2, gamma, phi, ktype=cv2.CV_32F)
kernel5 = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma*10, phi, ktype=cv2.CV_32F)
kernel6 = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi*(1/2), ktype=cv2.CV_32F)

kernel_list = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]
name_list = ['normal','sigma=6','theta=π/2', 'lambda=π/2', 'gamma=1', 'phi=π/2']

kernel_list = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]
name_list = ['normal', 'sigma=6', 'theta=π/2', 'lambda=π/2', 'gamma=1', 'phi=π/2']

def plot_parameter_graph():
    plt.figure(figsize=(6, 6))
    j=1
    for kernels in kernel_list:
        plt.subplot(2, 3, j)
        plt.imshow(kernels, cmap='rainbow')
        plt.xlabel(f'{name_list[j-1]}')
        j+=1

    plt.tight_layout()  
    #fig1=plt.gcf()
    #fig1.savefig(os.path.join(os.getcwd(), 'Gabor_Parameter_change.png'))
    #plt.close()
    plt.show()

plot_parameter_graph()

 
### The codes below are finer descriptions of Gabor paramters and their role
### The codes were adapted from the repository: 'https://github.com/biocomplab/ColourOpponencyGaborNet' 
sigmalist=[0.5]
blist=[1,1.8,2.6]
thetalist=[0,np.pi/4, 2*np.pi/4, 3*np.pi/4]
philist=[np.pi/2, 3*np.pi/2]
def calc_lambda(sigma, bandwidth):
    p = 2**bandwidth
    c = np.sqrt(np.log(2)/2)
    return sigma * np.pi / c  * (p - 1) / (p + 1)

x=2*5/5*10/10
y=(2*5*10)/(5*10)
if y==x:
    print(True)


#Parameter sigma and bandwidth:
#The bandwidth or sigma controls the overall size of the Gabor Envelope. For larger bandwidth
#the envelope increases allowing more stripes and with small bandwidth
#the envelope tightens. On increasing the sigma to 30 and 45, the number of stripes
#the Gabor function increases


j=0
for i in sigmalist:
    plt.subplot2grid((1, 5), (0, j))
    Lambda=calc_lambda(i,1.8)
    kernel= cv2.getGaborKernel((31,31), i, 0, Lambda, 0.5, np.pi/2)
    plt.imshow(kernel,cmap="rainbow")
    plt.xlabel("sigma="+str(i))
    j+=1
    
plt.show()



#Parameter Gamma:
#The aspect ratio or gamma controls the heights of the Gabor function.
#For very high aspect ratio the height becomes very small and for very small gamma value the heights becomes quite large
#On increasing the value the height becomes quite large.
#On increasing the value of gamma to 0.5 and 0.75,
#keeping other parameters unchanged, the height of the Gabor function reduces.

def main():
    j=0
    gammalist=[0.2,0.35,0.5,0.75,1]
    for i in gammalist:
        plt.subplot2grid((1, 5), (0, j))
        Lambda=calc_lambda(8,1.8)
        kernel= cv2.getGaborKernel((31,31), 8, 0, Lambda, i, np.pi/2)
        plt.imshow(kernel,cmap="rainbow")
        plt.xlabel("gamma="+str(i))
        j+=1
    plt.show()

if __name__ == "__main__":
    main()

#Parameter Theta:
#The theta controls the orientation of the Gabor function.
#The zero degree theta corresponds to the vertical position of the
#Gabor function.

def main():
    j=0
    thetalist=[0,1/5*np.pi,2/5*np.pi,3/5*np.pi,4/5*np.pi]
    for i in thetalist:
        plt.subplot2grid((1, 5), (0, j))
        Lambda=calc_lambda(8,1.8)
        kernel= cv2.getGaborKernel((31,31), 8, i, Lambda, 0.5, np.pi/2)
        plt.imshow(kernel,cmap="rainbow")
        plt.xlabel("theta="+str(round(i,3)))
        j+=1
    plt.show()

if __name__ == "__main__":
    main()
    
#Parameter Phi:
#The phase offsetr of the sinusoidal function

def main():
    j=0
    philist=[0,1/5*np.pi,2/5*np.pi,3/5*np.pi,4/5*np.pi]
    for i in philist:
        plt.subplot2grid((1, 5), (0, j))
        Lambda=calc_lambda(8,1.8)
        kernel= cv2.getGaborKernel((31,31), 8, 0, Lambda, 0.5, i)
        plt.imshow(kernel,cmap="rainbow")
        plt.xlabel("phi="+str(round(i,3)))
        j+=1
    plt.show()

if __name__ == "__main__":
    main()


def main():
    kszie = 31
    sigma = 8
    theta = np.pi/4
    def calc_lambda(sigma, bandwidth):
        p = 2**bandwidth
        c = np.sqrt(np.log(2)/2)
        return sigma * np.pi / c  * (p - 1) / (p + 1)
    b=1.6
    Lambda = calc_lambda(sigma,b)
    gamma = 0.5
    phi = np.pi/2
    kb=cv2.getGaborKernel((31, 31), sigma, theta, Lambda, gamma,phi, ktype=cv2.CV_32F)
    plt.imshow(kb, cmap='rainbow')
    plt.show()
    #kb.shape()
    X=np.arange(-30,31,1) #puts an array ranging from -30 to 31 with spaces 1
    Y=np.arange(-30,31,1) 
    print(f'Y = {Y}')
    print(f'X = {X}')
    print(X.shape[0])
    X,Y=np.meshgrid(X,Y) #This function turn these X's and Y's into a opposite type grids
    #X has same elements in rows while Y has same elements in columns
    Z=cv2.getGaborKernel((60, 60), sigma, theta, Lambda, gamma,phi, ktype=cv2.CV_32F)
    plt.imshow(Z)
    plt.show()
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)
    cset = ax.contour(X, Y, Z ,zdir='z', offset=-1, alpha=1,cmap=matplotlib.cm.coolwarm)
    cset = ax.contour(X, Y, Z ,zdir='x', offset=-30, cmap=matplotlib.cm.coolwarm)
    cset = ax.contour(X, Y, Z ,zdir='y', offset=30, cmap=matplotlib.cm.coolwarm)
    ax.set_zlabel("value")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.view_init(15, -45)
    #ax.set_xlim3d(-np.pi, 2*np.pi)
    #ax.set_ylim3d(, 3*np.pi)
    #ax.set_zlim3d(-np.pi, 2*np.pi)
    Z1=cv2.getGaborKernel((60, 60), sigma, theta, Lambda, gamma,phi, ktype=cv2.CV_32F)
    ax1=fig.add_subplot(1,2,1,projection="3d")
    sur=ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.95)
    cset = ax1.contour(X, Y, Z ,zdir='z', offset=-1, alpha=1,cmap=matplotlib.cm.coolwarm)
    cset = ax1.contour(X, Y, Z ,zdir='x', offset=-30, cmap=matplotlib.cm.coolwarm)
    cset = ax1.contour(X, Y, Z ,zdir='y', offset=30, cmap=matplotlib.cm.coolwarm)
    ax1.set_zlabel("value")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.view_init(0, -45)
    #fig.colorbar(sur,shrink=0.3)
    plt.show()

    

if __name__ == "__main__":
    main()


def main():
    
    sigmalist=[8]
    gammalist=[0.5]
    blist=[1,1.8,2.6]
    thetalist=[0,np.pi/4, 2*np.pi/4, 3*np.pi/4]
    philist=[np.pi/2, 3*np.pi/2]
    def calc_lambda(sigma, bandwidth):
        p = 2**bandwidth
        c = np.sqrt(np.log(2)/2)
        return sigma * np.pi / c  * (p - 1) / (p + 1)
    tl=[str(0), chr(960)+"/4", "2"+chr(960)+"/4","3"+chr(960)+"/4"]
    pl=[chr(960)+"/2","3"+chr(960)+"/2"]
    picsave=[]

    fig=plt.figure()

    plt.subplots_adjust(0, 0, 2, 1)

    od=0
    op=0
    i=0
    for b in blist:
        j=0
        op=0
        for theta in thetalist:
            for phi in philist:
                plt.subplot2grid((3, 8), (i, j))
                Lambda=calc_lambda(sigmalist[0],b)
                kernel= cv2.getGaborKernel((31,31), sigmalist[0], theta, Lambda, gammalist[0], phi)
                plt.xticks([])
                plt.yticks([])
                
                
                plt.imshow(kernel,cmap="rainbow")
                #plt.colorbar()
                j+=1
        i+=1
        od+=1
    #fig.colorbar()
    plt.show()

if __name__ == "__main__":
    main()







