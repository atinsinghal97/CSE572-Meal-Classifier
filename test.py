import os
import sys
import numpy as np
import pandas as pd

import math
import pywt

from sklearn.preprocessing import StandardScaler

# from sklearn import svm, model_selection
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import BernoulliNB, GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn import tree
# from sklearn.neighbors import KNeighborsClassifier

import scipy.stats
from scipy import fft
from scipy import signal

import pickle
import warnings

warnings.filterwarnings("ignore")

if (len(sys.argv)) == 1:
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    csvFile = path + os.sep + 'Test.csv'
    # csvFile = "/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/Test.csv"
    print("Using the default path for CSVFile: ", csvFile)
    if (input("Continue? (y/n)") == 'n'):
        print("Run the code as python <file-name.py> <Path-to-CSVFile>")
        print(
            "Eg: python test.py /Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/Test.csv")
        sys.exit(0)
elif (len(sys.argv)) == 2:
    csvFile = sys.argv[1]
else:
    print ("Error. Run the code as python <file-name.py> <Path-to-CSVFile>")
    print ("Eg: python test.py /Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/Test.csv")
    sys.exit(0)

def debug():
    print(FeatureMatrixTest)
    FeatureMatrixTest.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/Processed Data/FeatureMatrixTest.csv",
        index=None, header=True)


# Read CSV Files

testDataFiles = []
testDataFiles.append(csvFile)

testDataList = []
for file in testDataFiles:
    # if file == 'mealData4.csv':
    #     dfForDataLoad = pd.read_csv(directory+file, usecols=[*range(0,30)], skiprows=3)
    # elif file == 'mealData5.csv':
    #     dfForDataLoad = pd.read_csv(directory+file, usecols=[*range(0,30)], skiprows=1)
    # else:
    #     dfForDataLoad = pd.read_csv(directory + file, usecols=[*range(0, 30)])
    dfForDataLoad = pd.read_csv(file, names=list(range(0, 30)), usecols=[*range(0,30)])
    for row in dfForDataLoad.values:
        testDataList.append(row)

testData = pd.DataFrame(testDataList)

# Pre-Processing

# rowTest,colTest = testData.shape
# thresholdA1 = colTest*0.95
# thresholdA1 = int(thresholdA1)
# testData = testData.dropna(axis=0,thresh=thresholdA1)
testData.fillna(method='pad', inplace=True)
testData = testData.reset_index(drop=True)

# print(testData)

# testData.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/Processed Data/testData.csv", index=False, header=False)

FeatureMatrixTest = pd.DataFrame()


# FFT- Top 4 Instances

def FourierTransform(row):
    # FFTVal = abs(scipy.fftpack.fft(row))
    FFTVal = abs(fft(row))
    FFTVal.sort()
    return np.flip(FFTVal)[0:4]

FFT = pd.DataFrame()
FFT['FFTExtracted'] = testData.apply(lambda x: FourierTransform(x), axis=1)
FFT_updated = pd.DataFrame(FFT.FFTExtracted.tolist(), columns=['FFT1', 'FFT2', 'FFT3', 'FFT4'])
FeatureMatrixTest = FFT_updated
# print (FFT_updated)

# debug()


# RMS

def RMS(series):
    # Calculating Square
    sq=0
    n=0
    for entry in series:
        sq=sq+entry**2
        n += 1

    # Calculating Mean
    mean = sq/n

    # Calculatung Root
    root = math.sqrt(mean)

    return root

RMSdf = pd.DataFrame()
RMSdf['RMS'] = testData.apply(lambda x: RMS(x), axis=1)
# print (RMSdf)
FeatureMatrixTest = FeatureMatrixTest.merge(RMSdf, left_index=True, right_index=True)

# debug()


# PSD

def PSD(series):
    f, psd1=scipy.signal.welch(series)
    return float(psd1[0:1])

psddf = pd.DataFrame()
psddf['PSD'] = testData.apply(lambda x: PSD(x), axis=1)
# print (psddf)
# PSD_updated = pd.DataFrame(psddf.PSD.tolist(), columns=['PSD'])
# print (PSD_updated)
FeatureMatrixTest = FeatureMatrixTest.merge(psddf, left_index=True, right_index=True)

# debug()


# DWT- 7 instances

def DWT(series):
    ca, cb = pywt.dwt(series, 'haar')
    cat = pywt.threshold(ca, np.std(ca)/2, mode='soft')
    cbt = pywt.threshold(cb, np.std(cb)/2, mode='soft')

    signal = pywt.idwt(cat, cbt, 'haar')

    DWT8 = ca[:,:-8] #sorted in Ascending

    return DWT8

DWTdf = pd.DataFrame()
DWTdf=DWT(testData)
DWTdf = pd.DataFrame(DWTdf.tolist(), columns=['DWT1', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7'])
# print (DWTdf)
FeatureMatrixTest = FeatureMatrixTest.merge(DWTdf, left_index=True, right_index=True)

# debug()


# Dropping Unnecessary features
# FeatureMatrixMeal = FeatureMatrixMeal.drop(columns=['FFT2', 'FFT3', 'FFT4', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7'])
# FeatureMatrixNoMeal = FeatureMatrixNoMeal.drop(columns=['FFT2', 'FFT3', 'FFT4', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7'])


# Normalize

FeatureMatrixTest = StandardScaler().fit_transform(FeatureMatrixTest)

# FeatureList = ['FFT1', 'RMS', 'PSD', 'DWT1']
FeatureList = ['FFT1', 'FFT2', 'FFT3', 'FFT4', 'RMS', 'PSD', 'DWT1', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7']
FeatureMatrixTest = pd.DataFrame(data=FeatureMatrixTest, index=None, columns=FeatureList)


# Getting X, y ready
Xdf = pd.concat([FeatureMatrixTest], ignore_index=True)
# ydf = pd.concat([pd.DataFrame({'Class': [1]*len(FeatureMatrixMeal.index)}), pd.DataFrame({'Class': [0]*len(FeatureMatrixNoMeal.index)})], ignore_index=True)

X = Xdf.to_numpy()
# y = ydf.to_numpy()

# Load Model
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
modelName = path + os.sep + '1217358454.pkl'
finalClassifier = pickle.load(open(modelName, 'rb'))
output = finalClassifier.predict(FeatureMatrixTest)

print (output)

# print (sum(output))
# print (len(output))

# debug()