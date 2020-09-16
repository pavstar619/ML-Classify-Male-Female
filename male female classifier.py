#import enssential dependencies from sklearn
from sklearn import ensemble
from sklearn import svm
import pandas as pd 

#[height, weight, shoe size]
X = [[181,80,44], [177, 70, 43], [160, 60, 38], [154, 54,37], 
	 [166,65,40], [190,90,47], [175,64,39], [177,70,40], [159, 55,39], [171,75,42], [181,85,43], [220,115,53], [210,110,51], [230,120,55]]

#gender 1 is MALE, 0 is FEMALE
Y =  [1, 0, 0, 0, 1, 1,
	  1, 0,1, 0, 1, 2, 2, 2]

""" df = pd.read_excel('test.xlsx', 'sheet1')
X = df['Features'].values.tolist()
Y = df['Gender'].values.tolist()
print(X)
print(Y) """

#create variables with classifiers
randomForest = ensemble.RandomForestClassifier()
svm = svm.SVC(gamma='auto')
adaBoost = ensemble.AdaBoostClassifier()

#fit them
randomForest = randomForest.fit(X,Y)
svm = svm.fit(X,Y)
adaBoost = adaBoost.fit(X,Y)

#Take inputs
print('Write inputs cm, kg, size')
a, b, c = input().split()

#create prediction variables for results
predictionForRandomForest = randomForest.predict([[a,b,c]])
predictionForSVM = svm.predict([[a,b,c]])
predictionForAdaBoost = adaBoost.predict([[a,b,c]])


#print all the results
print("2 for GIANT, 1 for MALE and 0 for FEMALE")
print("RandomForest: ", predictionForRandomForest)
print("SVM: ", predictionForSVM)
print("AdaBoost: ", predictionForAdaBoost)
