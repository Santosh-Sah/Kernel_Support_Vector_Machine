# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:39:09 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from KernelSupportVectorMachineUtils import (readKernelSupportVectorMachineYTest, readKernelSupportVectorMachineYPred)

"""

calculating KernalSupportVectorMachine confussion matrix

"""
def testKernelSupportVectorMachineConfussionMatrix():
    
    y_test = readKernelSupportVectorMachineYTest()
    y_pred = readKernelSupportVectorMachineYPred()
    
    kernelSupportVectorMachineConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(kernelSupportVectorMachineConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[57  1]
    [ 6 16]]
    
    """
"""
calculating accuracy score

"""

def testKernelSupportVectorMachineAccuracy():
    
    y_test = readKernelSupportVectorMachineYTest()
    y_pred = readKernelSupportVectorMachineYPred()
    
    kernelSupportVectorMachineConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(kernelSupportVectorMachineConfussionAccuracy) #.9125%

"""
calculating classification report

"""

def testKernelSupportVectorMachineClassificationReport():
    
    y_test = readKernelSupportVectorMachineYTest()
    y_pred = readKernelSupportVectorMachineYPred()
    
    kernelSupportVectorMachineConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(kernelSupportVectorMachineConfussionClassificationReport)
    
    """
               precision    recall  f1-score   support

          0       0.90      0.98      0.94        58
          1       0.94      0.73      0.82        22

avg / total       0.91      0.91      0.91        80
    """
    
if __name__ == "__main__":
    #testKernelSupportVectorMachineConfussionMatrix()
    #testKernelSupportVectorMachineAccuracy()
    testKernelSupportVectorMachineClassificationReport()