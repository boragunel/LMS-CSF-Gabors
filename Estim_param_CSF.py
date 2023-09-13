

    ### The code in this script was suggested for calculating appropriate parameter values using 'Contrast Sensitivity Functions' (CSF's)
    ### for the colour-opponent Gabor filter to make it more biologically plausible
    ### The functions mentioned were adapted from Kathy Mullen's research (2017):
    ### 'A Normative Data Set for the Clinical Assessment of Achromatic and Chromatic Contrast Sensitivity Using a qCSF Approach'


import matplotlib.pyplot as plt
import math
import numpy as np




'''
B=3
k=math.log10(2)
BI=math.log10(2*B)
Y=100
fmax=5
#delta=2.39-0.24
param_list=[]
freq_list=[]
param_dict={}

def main():

    def S(f):
        Gain = math.pow(((math.log10(f)-math.log10(fmax))/(BI/2)),2)
        Gain = math.log10(Y)-(Gain*k)
        if f<fmax and Gain<(math.log10(Y)-f):
            G=math.log10(Y)-f
            return G
        else:
            return Gain

    def param_test():
        x=0.24
        for param in range(232):
            x+=0.1
            param_list.append(S(x))
            freq_list.append(x)
            
        return param_list, freq_list
        
    def plot_param():
        plt.plot(freq_list, param_list)
        plt.xlabel('Frequency(cycles/degree)')
        plt.ylabel('Gain')
        plt.show()

    param_test()
    plot_param()
        




if __name__ == "__main__":
    main()
'''

### The following functions are calcuating appropriate bandwidth values by best eye estimate
### for Achromatic, Isoluminant Red-Green and Isoluminant Blue-Yellow gratings

def Achromatic():
    k=math.log10(2)
    Y=34.7
    fmax=1.63
    param_list=[]
    freq_list=[]
    param_dict={}
    
    def S(f,B):
        BI=math.log10(2*B)
        Gain = math.pow(((math.log10(f)-math.log10(fmax))/(BI/2)),2)
        Gain = math.log10(Y)-(Gain*k)
        #if f<fmax and Gain<(math.log10(Y)-f):
        #    G=math.log10(Y)-f
        #    return G
        
        return Gain
    
    def param_iter(): 
        b_list=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
        for b in b_list:
            plt.figure()
            x=0.24
            param_list=[]
            freq_list=[]
            for param in range(232):
                x+=0.1
                param_list.append(S(x,b))
                freq_list.append(math.log10(x))
            plt.plot(freq_list, param_list)
            plt.xlabel('Frequency(cycles/degree)')
            plt.ylabel('Gain')
            plt.title(f'bandwidth = {b}')
            plt.show()
    
    param_iter()
                
Achromatic()
                            
        


        
def Isoluminant_RG():
    k=math.log10(2)
    Y=204.6
    fmax=0.58
    param_list=[]
    freq_list=[]
    param_dict={}
    
    def S(f,B):
        BI=math.log10(2*B)
        Gain = math.pow(((math.log10(f)-math.log10(fmax))/(BI/2)),2)
        Gain = math.log10(Y)-(Gain*k)
        #if f<fmax and Gain<(math.log10(Y)-f):
        #    G=math.log10(Y)-f
        #    return G
        
        return Gain
    
    def param_iter():
        b_list=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
        
        for b in b_list:
            plt.figure()
            x=0.24
            param_list=[]
            freq_list=[]
            for param in range(232):
                x+=0.1
                param_list.append(S(x,b))
                freq_list.append(math.log10(x))
            plt.plot(freq_list, param_list)
            plt.xlabel('Frequency(cycles/degree)')
            plt.ylabel('Gain')
            plt.title(f'bandwidth = {b}')
            plt.show()
    
    param_iter()

Isoluminant_RG()
    
def Isoluminant_BY():
    k=math.log10(2)
    Y=28.05
    fmax=0.49
    param_list=[]
    freq_list=[]
    param_dict={}
    
    def S(f,B):
        BI=math.log10(2*B)
        Gain = math.pow(((math.log10(f)-math.log10(fmax))/(BI/2)),2)
        Gain = math.log10(Y)-(Gain*k)
        #if f<fmax and Gain<(math.log10(Y)-f):
        #    G=math.log10(Y)-f
        #    return G
        
        return Gain
    
    def param_iter():
        b_list=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
        
        for b in b_list:
            plt.figure()
            x=0.24
            param_list=[]
            freq_list=[]
            for param in range(232):
                x+=0.1
                param_list.append(S(x,b))
                freq_list.append(math.log10(x))
            print(f'the maximum value = {max(param_list)}, the minimum value is = {min(param_list)} for b = {b}')
            '''
            plt.plot(freq_list, param_list)
            plt.xlabel('Frequency(cycles/degree)')
            plt.ylabel('Gain')
            plt.title(f'bandwidth = {b}')
            plt.show()
            '''
    
    param_iter()

Isoluminant_BY()


### Having extracted the appropriate bandwidth values, the below functions are for calculating the appropraite sigma parameters
        

fmax_list=[1.63, 0.49, 0.58]

b_list=[2,2.5,3,3.5,4,4.5,5]
sigma_list=[]



def calc_lambda(self, sigma, bandwidth):            
    p = 2**bandwidth                             
    c = np.sqrt(np.log(2)/2)
    return sigma * np.pi / c  * (p - 1) / (p + 1)

def cal_sigma(bw, fmax):
    a=1/math.pi
    b=1/(math.sqrt(2 / (math.log(2,math.e))))
    c=((2**bw) + 1) / ((2**bw) - 1)
    d= 1 / fmax
    lamda=a*b*c*d
    return 1/lamda


def calculate_sigmas():
    for b in b_list:
        for freq in fmax_list:
            x=cal_sigma(b,freq)
            rounded_x = round(x, 2)  # Round to two decimal places
            formatted_x = "{:.2f}".format(rounded_x)
            sigma_list.append(formatted_x)
    return print(sigma_list)

calculate_sigmas()



### Please note that the methods were adapted as bandiwdth parameter of Gabor functions
### were theorised to be aligning with the bandwidth values of the CSF's
### There is room for further development and alternative hypothesis for estimating appropriate parameter values 

