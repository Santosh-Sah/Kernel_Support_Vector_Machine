# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:48:29 2020

@author: Santosh Sah
"""

import pandas as pd
from KernelSupportVectorMachineUtils import readKernelSupportVectorMachineModel, readKernelSupportVectorMachineStandardScaler

def predict():
    
    kernelSupportVectorMachine = readKernelSupportVectorMachineModel()
    kernelSupportVectorMachineStandardScaler = readKernelSupportVectorMachineStandardScaler()
    
    inputValue = [[26, 1000]]
    inputValueDataframe = pd.DataFrame(kernelSupportVectorMachineStandardScaler.transform(inputValue))
    
    predictedValue = kernelSupportVectorMachine.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()