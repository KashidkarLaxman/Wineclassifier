from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import datasets
def knn() :
	data=datasets.load_wine()
	feature_names=data.feature_names
	target=data.target

	
	#feature_names=pd.DataFrame(data,columns=['alcohol','malic acid','Ash','Alcanity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoids phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'])
	#target=pd.DataFrame(data,columns=['Class'])
	train_data,test_data,train_target,test_target=train_test_split(feature_names,target,test_size=0.3)
	
	classifier=KNeighborsClassifier()
	print(feature_names)
	try : 
		classifier.fit(train_data,train_target)
	except Exception as e :
		print(e)
	prediction=classifier.predict(test_data)
	print(prediction)
	

def main() :
	print("Wine predictor")
	knn()

if __name__ =="__main__" :
	main()