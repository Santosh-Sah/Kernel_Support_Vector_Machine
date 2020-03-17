# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:01:45 2020

@author: Santosh Sah
"""

from KernelSupportVectorMachineUtils import (readKernelSupportVectorMachineXTest, readKernelSupportVectorMachineModel,
                                     saveKernelSupportVectorMachineYPred, readKernelSupportVectorMachineStandardScaler)

"""
test the model on testing dataset
"""
def testKernelSupportVectorMachineModel():
    
    X_test = readKernelSupportVectorMachineXTest()
    kernelSupportVectorMachineStandardScaler = readKernelSupportVectorMachineStandardScaler()
    X_test = kernelSupportVectorMachineStandardScaler.transform(X_test)
    
    kernelSupportVectorMachineModel = readKernelSupportVectorMachineModel()
    
    y_pred = kernelSupportVectorMachineModel.predict(X_test)
    saveKernelSupportVectorMachineYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testKernelSupportVectorMachineModel()