# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:02:07 2020

@author: Santosh Sah
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from KernelSupportVectorMachineUtils import (saveKernelSupportVectorMachineModel, readKernelSupportVectorMachineXTrain, readKernelSupportVectorMachineYTrain,
                                     saveKernelSupportVectorMachineStandardScaler)

"""
Train KernelSupportVectorMachine model 
"""
def trainKernelSupportVectorMachineModel():
    
    kernelSupportVectorMachineStandardScalar = StandardScaler()
    
    X_train = readKernelSupportVectorMachineXTrain()
    y_train = readKernelSupportVectorMachineYTrain()
    
    kernelSupportVectorMachineStandardScalar.fit(X_train)
    saveKernelSupportVectorMachineStandardScaler(kernelSupportVectorMachineStandardScalar)
    
    X_train = kernelSupportVectorMachineStandardScalar.transform(X_train)
    
    kernelSupportVectorMachine = SVC(kernel = "rbf", random_state = 1234)
    kernelSupportVectorMachine.fit(X_train, y_train)
    
    saveKernelSupportVectorMachineModel(kernelSupportVectorMachine)

if __name__ == "__main__":
    trainKernelSupportVectorMachineModel()