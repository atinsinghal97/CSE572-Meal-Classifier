# Meal-Classifier
Info in the `todo.pdf` file.


How to compile code?
--------------------
This code uses Python3. The python code file imports the following libraries:
1. numpy
2. pandas
3. PyWavelets (pywt)
4. sklearn
5. scipy
6. math
7. os
8. sys
9. pickle
10. warnings


They can be installed by using the following commands:
1. pip install numpy
2. pip install pandas
3. pip install PyWavelets
4. pip install sklearn
5. pip install scipy


Run the training code as "python train.py <Path-to-DataFolder>"
Eg: python train.py /Users/Documents/Projects/MealNoMealData/

Run the testing code as "python test.py <Path-to-CSVFile>"
Eg: python test.py /Users/Documents/Projects/Test.csv

Note# If your system has both Python2 & Python3, use pip3 instead of pip & python3 instead of python to compile.


Contents of Folder Submitted
-----------------------------

Files: 
1. train.py- PYTHON Code File for Training the model
2. test.py- PYTHON Code File for Testing the model
3. readme.txt

Folders:
1. MealNoMealData- This is the original data folder that was provided.
2. model- Contains the PKL model file.


NOTE# 
Sample Output:
{'fit_time': array([0.05648589, 0.05658698, 0.05727792]), 'score_time': array([0.00545979, 0.00511193, 0.00732422]), 'test_accuracy': array([0.72297297, 0.70748299, 0.67346939]), 'test_recall': array([0.73972603, 0.60526316, 0.6       ]), 'test_f1': array([0.72483221, 0.68148148, 0.65217391]), 'test_precision': array([0.71052632, 0.77966102, 0.71428571])}

Interpreted as:
Accuracy = [0.72297297, 0.70748299, 0.67346939]
Recall = [0.73972603, 0.60526316, 0.6       ]
F1 = [0.72483221, 0.68148148, 0.65217391]
Precision = [0.71052632, 0.77966102, 0.71428571]

3 results as I used k=3 in k-Fold
