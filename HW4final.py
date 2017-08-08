from PIL import Image
import numpy as np
import pandas as pd
from glob import glob
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

def imToArray (path):

    # read picture and convert it to black and white
	pic = Image.open(path).convert('L')
	# resize the picture to the same size, reduce dimension
	pic = pic.resize((100,50))
	# convert picture to nparray
	pic = np.array(pic)

	# convert numpy.ndarray into vector
	x1 = np.array(pic.flatten())
	return x1;


topic = np.array(["airplanes", "Bonsai", "Faces", "Leopards", "Motorbikes"])

filetrain = np.array(['/home/andy/ML/homework data/airplanes/Train/image*.jpg',
	'/home/andy/ML/homework data/Bonsai/train/image*.jpg',
	'/home/andy/ML/homework data/Faces/Train/image*.jpg',
	'/home/andy/ML/homework data/Leopards/Train/image*.jpg',
	'/home/andy/ML/homework data/Motorbikes/Train/image*.jpg'])

filetest = np.array(['/home/andy/ML/homework data/airplanes/Test/image*.jpg',
	'/home/andy/ML/homework data/Bonsai/test/image*.jpg',
	'/home/andy/ML/homework data/Faces/Test/image*.jpg',
	'/home/andy/ML/homework data/Leopards/Test/image*.jpg',
	'/home/andy/ML/homework data/Motorbikes/Test/image*.jpg'])

data = np.array(['/home/andy/ML/homework data/airplanes_data.csv',
	'/home/andy/ML/homework data/Bonsai_data.csv',
	'/home/andy/ML/homework data/Faces_data.csv',
	'/home/andy/ML/homework data/Leopards_data.csv',
	'/home/andy/ML/homework data/Motorbikes_data.csv'])

for j in range(0,len(filetrain)):
	# read all the image names from Train folder
	filelist_train = glob(filetrain[j])

	# make the filelist_train into a pandas dataframe
	name_train = pd.Series(filelist_train)
	for i in range(0,len(filelist_train)):
		name_train[i] = name_train[i].split('/')[-1]
	name_train = pd.DataFrame(name_train)
	name_train.columns = ["name"]

	# read all the train images
	x = list()
	for i in range(0,len(filelist_train)):
		item = imToArray(filelist_train[i])
		x.append(item)
	x = np.array(x)

	# read the label of each picture
	y = pd.read_csv(data[j])

	# convert y into pandas dataframe ane make the name of the file match the filelist_train
	for i in range(0,len(y)):
		y.ix[i,0] = y.ix[i,0].split('_',1)[-1]

	# left join these two table because I only want the label of train data set
	ynew = pd.merge(name_train, y, on='name')
	label = np.ravel(ynew['final'])

	# read the test dataset
	filelist_test = glob(filetest[j])

	# make the filelist_test into a pandas dataframe
	name_test = pd.Series(filelist_test)
	for i in range(0,len(filelist_test)):
		name_test[i] = name_test[i].split('/')[-1]
	name_test = pd.DataFrame(name_test)
	name_test.columns = ["name"]

	# read all the test images
	xtest = list()
	for i in range(0,len(filelist_test)):
		item = imToArray(filelist_test[i])
		xtest.append(item)
	xtest = np.array(xtest)

	# get the labels for test data det
	ytest = pd.merge(name_test, y, on='name')
	labeltest = np.ravel(ytest['final'])

	# use PCA to reduce dimension
	n = np.array([161,49,105,51,100])
	pca = PCA(n_components = n[j], svd_solver='full')
	x1 = pca.fit(x)
	x2 = x1.transform(x)
	xtest1 = x1.transform(xtest)

	# fit them different models
	lgmodel = LogisticRegression()
	lgmodel = lgmodel.fit(x2,label)
	svmmodel = svm.LinearSVC()
	svmmodel = svmmodel.fit(x2,label)
	ksvmmodel = svm.SVC(kernel = 'rbf')
	ksvmmodel = svmmodel.fit(x2,label)

	# adaline has no exsiting packages
	lnmodel = LinearRegression()
	lnmodel = lnmodel.fit(x2,label)
	lnpredict = lnmodel.predict(x2)
	count = 0
	for i in range(0,len(lnpredict)):
		if lnpredict[i] > 0:
			lnpredict[i] = 1
		if lnpredict[i] <= 0:
			lnpredict[i] = 0
	for i in range(0,len(lnpredict)):
		if lnpredict[i] == label[i]:
			count = count + 1
	ada_train_accuracy = float(count)/float(len(lnpredict))

	lntestpredict = lnmodel.predict(xtest1)
	count = 0
	for i in range(0,len(lntestpredict)):
		if lntestpredict[i] > 0:
			lntestpredict[i] = 1
		if lntestpredict[i] <= 0:
			lntestpredict[i] = 0
	for i in range(0,len(lntestpredict)):
		if lntestpredict[i] == labeltest[i]:
			count = count + 1
	ada_test_accuracy = float(count)/float(len(lntestpredict))

	# print the result
	print(topic[j]+":")
	print("Logistics Accuracy for tarin dataset:"+str(lgmodel.score(x2,label)))
	print("LOgistics Accuracy for test dataset:"+str(lgmodel.score(xtest1,labeltest)))
	print("SVM Accuracy for tarin dataset:"+str(svmmodel.score(x2,label)))
	print("SVM Accuracy for test dataset:"+str(svmmodel.score(xtest1,labeltest)))
	print("KSVM Accuracy for tarin dataset:"+str(ksvmmodel.score(x2,label)))
	print("KSVM Accuracy for test dataset:"+str(ksvmmodel.score(xtest1,labeltest)))
	print("ada Accuracy for tarin dataset:"+str(ada_train_accuracy))
	print("ada Accuracy for test dataset:"+str(ada_test_accuracy))
	print("\n")