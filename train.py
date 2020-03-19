import os
import sys
import numpy as np
import pandas as pd

import math
import pywt

from sklearn.preprocessing import StandardScaler

from sklearn import svm, model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

import scipy.stats
from scipy import fft
from scipy import signal

import pickle
import warnings

warnings.filterwarnings("ignore")

if (len(sys.argv)) == 1:
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    directory = path + os.sep + 'MealNoMealData' + os.sep
    # directory = "/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/MealNoMealData/"
    print ("Using the default path for Data Folder: ", directory)
    if (input ("Continue? (y/n)") == 'n'):
        print("Run the code as python <file-name.py> <Path-to-DataFolder>")
        print(
            "Eg: python train.py /Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/MealNoMealData/")
        print ("Make sure to use the os separator at the end. It's ", os.sep, " for your OS.")
        sys.exit(0)
elif (len(sys.argv)) == 2:
    directory = sys.argv[1]
    if directory[-1]!=os.sep:
        directory = directory + os.sep
else:
    print ("Error. Run the code as python <file-name.py> <Path-to-DataFolder>")
    print ("Eg: python train.py /Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/MealNoMealData/")
    print ("Make sure to use the os separator at the end. It's ", os.sep, " for your OS.")
    sys.exit(0)

def debug():
    print(FeatureMatrixMeal)
    print(FeatureMatrixNoMeal)
    FeatureMatrixMeal.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/Processed Data/FeatureMatrixMeal.csv",
        index=None, header=True)
    FeatureMatrixNoMeal.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/Processed Data/FeatureMatrixNoMeal.csv",
        index=None, header=True)

# FeaturesList = list()

dfForDataLoad = pd.DataFrame()
mealFiles = ['mealData1.csv', 'mealData2.csv', 'mealData3.csv', 'mealData4.csv', 'mealData5.csv']
noMealFiles = ['Nomeal1.csv', 'Nomeal2.csv', 'Nomeal3.csv', 'Nomeal4.csv', 'Nomeal5.csv']

# Read CSV Files

mealList = []
for file in mealFiles:
    # if file == 'mealData4.csv':
    #     dfForDataLoad = pd.read_csv(directory+file, usecols=[*range(0,30)], skiprows=3)
    # elif file == 'mealData5.csv':
    #     dfForDataLoad = pd.read_csv(directory+file, usecols=[*range(0,30)], skiprows=1)
    # else:
    #     dfForDataLoad = pd.read_csv(directory + file, usecols=[*range(0, 30)])
    dfForDataLoad = pd.read_csv(directory + file, names=list(range(0, 30)), usecols=[*range(0,30)])
    for row in dfForDataLoad.values:
        mealList.append(row)

MealAll = pd.DataFrame(mealList)

noMealList = []
for file in noMealFiles:
    # dfForDataLoad = pd.read_csv(directory + file, names=list(range(0,30)), usecols=[*range(0,30)])
    dfForDataLoad = pd.read_csv(directory + file, usecols=[*range(0, 30)])
    # NoMealAll = pd.concat([NoMealAll, dfForDataLoad], ignore_index=True)
    for row in dfForDataLoad.values:
        noMealList.append(row)

NoMealAll = pd.DataFrame(noMealList)


# Pre-Processing

rowMeal,colMeal = MealAll.shape
thresholdA1 = colMeal*0.95
thresholdA1 = int(thresholdA1)
MealAll = MealAll.dropna(axis=0,thresh=thresholdA1)
MealAll.fillna(method='pad', inplace=True)
MealAll = MealAll.reset_index(drop=True)

rowNoMeal,colNoMeal = NoMealAll.shape
thresholdA2 = colNoMeal*0.95
thresholdA2 = int(thresholdA2)
NoMealAll = NoMealAll.dropna(axis=0,thresh=thresholdA2)
NoMealAll.fillna(method='pad', inplace=True)
NoMealAll = NoMealAll.reset_index(drop=True)

# print(MealAll)
# print(NoMealAll)

# MealAll.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/Processed Data/MealAll.csv", index=False, header=False)
# NoMealAll.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 2/Processed Data/NoMealAll.csv", index=False, header=False)

FeatureMatrixMeal = pd.DataFrame()
FeatureMatrixNoMeal = pd.DataFrame()


# FFT- Top 4 Instances

def FourierTransform(row):
    # FFTVal = abs(scipy.fftpack.fft(row))
    FFTVal = abs(fft(row))
    FFTVal.sort()
    return np.flip(FFTVal)[0:4]

FFT = pd.DataFrame()
FFT['FFTExtracted'] = MealAll.apply(lambda x: FourierTransform(x), axis=1)
FFT_updated = pd.DataFrame(FFT.FFTExtracted.tolist(), columns=['FFT1', 'FFT2', 'FFT3', 'FFT4'])
FeatureMatrixMeal = FFT_updated
# print (FFT_updated)

FFT = pd.DataFrame()
FFT['FFTExtracted'] = NoMealAll.apply(lambda x: FourierTransform(x), axis=1)
FFT_updated = pd.DataFrame(FFT.FFTExtracted.tolist(), columns=['FFT1', 'FFT2', 'FFT3', 'FFT4'])
FeatureMatrixNoMeal = FFT_updated
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
RMSdf['RMS'] = MealAll.apply(lambda x: RMS(x), axis=1)
# print (RMSdf)
FeatureMatrixMeal = FeatureMatrixMeal.merge(RMSdf, left_index=True, right_index=True)

RMSdf = pd.DataFrame()
RMSdf['RMS'] = NoMealAll.apply(lambda x: RMS(x), axis=1)
# print (RMSdf)
FeatureMatrixNoMeal = FeatureMatrixNoMeal.merge(RMSdf, left_index=True, right_index=True)

# debug()


# PSD

def PSD(series):
    f, psd1=scipy.signal.welch(series)
    return float(psd1[0:1])

psddf = pd.DataFrame()
psddf['PSD'] = MealAll.apply(lambda x: PSD(x), axis=1)
# print (psddf)
# PSD_updated = pd.DataFrame(psddf.PSD.tolist(), columns=['PSD'])
# print (PSD_updated)
FeatureMatrixMeal = FeatureMatrixMeal.merge(psddf, left_index=True, right_index=True)

psddf = pd.DataFrame()
psddf['PSD'] = NoMealAll.apply(lambda x: PSD(x), axis=1)
# print (psddf)
# PSD_updated = pd.DataFrame(psddf.PSD.tolist(), columns=['PSD'])
# print (PSD_updated)
FeatureMatrixNoMeal = FeatureMatrixNoMeal.merge(psddf, left_index=True, right_index=True)

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
DWTdf=DWT(MealAll)
DWTdf = pd.DataFrame(DWTdf.tolist(), columns=['DWT1', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7'])
# print (DWTdf)
FeatureMatrixMeal = FeatureMatrixMeal.merge(DWTdf, left_index=True, right_index=True)

DWTdf = pd.DataFrame()
DWTdf=DWT(NoMealAll)
DWTdf = pd.DataFrame(DWTdf.tolist(), columns=['DWT1', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7'])
# print (DWTdf)
FeatureMatrixNoMeal = FeatureMatrixNoMeal.merge(DWTdf, left_index=True, right_index=True)

# debug()


# Dropping Unnecessary features
# FeatureMatrixMeal = FeatureMatrixMeal.drop(columns=['FFT2', 'FFT3', 'FFT4', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7'])
# FeatureMatrixNoMeal = FeatureMatrixNoMeal.drop(columns=['FFT2', 'FFT3', 'FFT4', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7'])


# Normalize

FeatureMatrixMeal = StandardScaler().fit_transform(FeatureMatrixMeal)
FeatureMatrixNoMeal = StandardScaler().fit_transform(FeatureMatrixNoMeal)

# FeatureList = ['FFT1', 'RMS', 'PSD', 'DWT1']
FeatureList = ['FFT1', 'FFT2', 'FFT3', 'FFT4', 'RMS', 'PSD', 'DWT1', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7']
FeatureMatrixMeal = pd.DataFrame(data=FeatureMatrixMeal, index=None, columns=FeatureList)
FeatureMatrixNoMeal = pd.DataFrame(data=FeatureMatrixNoMeal, index=None, columns=FeatureList)


# Getting X, y ready
Xdf = pd.concat([FeatureMatrixMeal, FeatureMatrixNoMeal], ignore_index=True)
ydf = pd.concat([pd.DataFrame({'Class': [1]*len(FeatureMatrixMeal.index)}), pd.DataFrame({'Class': [0]*len(FeatureMatrixNoMeal.index)})], ignore_index=True)

X = Xdf.to_numpy()
y = ydf.to_numpy()

# Train
scoring ={'precision', 'recall', 'f1', 'accuracy'}

k_fold = KFold(n_splits=3, shuffle=True, random_state=0)

# clf = BernoulliNB()
# # clf.fit(X, y)
# print ('BNB: ', cross_val_score(clf, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf, X=X, y=y, cv=k_fold, scoring=scoring))
#
# clf2 = svm.SVC()
# print ('SVM: ', cross_val_score(clf2, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf2, X=X, y=y, cv=k_fold, scoring=scoring))
#
# clf3 = LogisticRegression(random_state=0)
# print ('LR: ', cross_val_score(clf3, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf3, X=X, y=y, cv=k_fold, scoring=scoring))
#
# clf4 = GaussianNB()
# print ('GNB: ', cross_val_score(clf4, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf4, X=X, y=y, cv=k_fold, scoring=scoring))
#
# clf5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 3), random_state=1)
# print ('MLP: ', cross_val_score(clf5, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf5, X=X, y=y, cv=k_fold, scoring=scoring))
#
# clf6 = AdaBoostClassifier(n_estimators=50, random_state=0)
# print ('Ada: ', cross_val_score(clf6, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf6, X=X, y=y, cv=k_fold, scoring=scoring))

clf7 = RandomForestClassifier(n_estimators=50)
# print ('RF: ', cross_val_score(clf7, X, y, cv=k_fold, n_jobs=1))
print (model_selection.cross_validate(estimator=clf7, X=X, y=y, cv=k_fold, scoring=scoring))

# clf8 = tree.DecisionTreeClassifier()
# print ('DT: ', cross_val_score(clf8, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf8, X=X, y=y, cv=k_fold, scoring=scoring))
#
# clf9 = KNeighborsClassifier(n_neighbors=5)
# print ('kNN: ', cross_val_score(clf9, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf9, X=X, y=y, cv=k_fold, scoring=scoring))


finalClassifier = clf7
finalClassifier.fit(X, y)
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
modelName = path + os.sep + '1217358454.pkl'
pickle.dump(finalClassifier, open(modelName, 'wb'))

# debug()