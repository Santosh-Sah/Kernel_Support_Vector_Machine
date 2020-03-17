# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:09:12 2020

@author: Santosh Sah
"""

"""
importing the libraries
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importKernelSupportVectorMachineDataset(kernelSupportVectorMachineDatasetFileName):
    
    kernelSupportVectorMachineDataset = pd.read_csv(kernelSupportVectorMachineDatasetFileName)
    X = kernelSupportVectorMachineDataset.iloc[:, [2, 3]].values
    y = kernelSupportVectorMachineDataset.iloc[:, 4].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveKernelSupportVectorMachineStandardScaler(kernelSupportVectorMachineStandardScalar):
    
    #Write KernelSupportVectorMachineStandardScaler in a picke file
    with open("KernelSupportVectorMachineStandardScaler.pkl",'wb') as KernelSupportVectorMachineStandardScaler_Pickle:
        pickle.dump(kernelSupportVectorMachineStandardScalar, KernelSupportVectorMachineStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save KernelSupportVectorMachineModel as a pickle file.
"""
def saveKernelSupportVectorMachineModel(kernelSupportVectorMachineModel):
    
    #Write KernelSupportVectorMachineModel as a picke file
    with open("KernelSupportVectorMachineModel.pkl",'wb') as KernelSupportVectorMachineModel_Pickle:
        pickle.dump(kernelSupportVectorMachineModel, KernelSupportVectorMachineModel_Pickle, protocol = 2)

"""
read KernelSupportVectorMachineStandardScalar from pickel file
"""
def readKernelSupportVectorMachineStandardScaler():
    
    #load KernelSupportVectorMachineStandardScaler object
    with open("KernelSupportVectorMachineStandardScaler.pkl","rb") as KernelSupportVectorMachineStandardScaler:
        kernelSupportVectorMachineStandardScalar = pickle.load(KernelSupportVectorMachineStandardScaler)
    
    return kernelSupportVectorMachineStandardScalar

"""
read KernelsupportVectorMachineModel from pickle file
"""
def readKernelSupportVectorMachineModel():
    
    #load KernelSupportVectorMachineModel model
    with open("KernelSupportVectorMachineModel.pkl","rb") as KernelSupportVectorMachineModel:
        kernelSupportVectorMachineModel = pickle.load(KernelSupportVectorMachineModel)
    
    return kernelSupportVectorMachineModel

"""
read X_train from pickle file
"""
def readKernelSupportVectorMachineXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readKernelSupportVectorMachineXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readKernelSupportVectorMachineYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readKernelSupportVectorMachineYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveKernelSupportVectorMachineYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readKernelSupportVectorMachineYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred