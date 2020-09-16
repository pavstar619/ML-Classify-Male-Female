#import enssential dependencies from sklearn
from  sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import svm

#[height, weight, shoe size]
X = [[181,80,44], [177, 70, 43], [160, 60, 38], [154, 54,37], 
	 [166,65,40], [190,90,47], [175,64,39], [177,70,40], [159, 55,39], [171,75,42], [181,85,43]]

#gender 1 is MALE, 0 is FEMALE
Y =  [1, 0, 0, 0, 1, 1,
	  1, 0,1, 0, 1]

#create variables with classifiers
randomForest = ensemble.RandomForestClassifier()
svm = svm.SVC(gamma='auto')

#fit them
randomForest = randomForest.fit(X,Y)
svm = svm.fit(X,Y)

#Take inputs
print('Write inputs')
a, b, c = input().split()

#create prediction variables for results
predictionForRandomForest = randomForest.predict([[a,b,c]])
predictionForSVM = svm.predict([[a,b,c]])


#print all the results
print("RandomForest: ", predictionForRandomForest)
print("SVM: ", predictionForSVM)
