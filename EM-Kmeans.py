from sklearn import datasets 
iris = datasets.load_iris()  # loading dataset

import pandas as pd 
X = pd.DataFrame(iris.data) # initializing rows and columns
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] 
y = pd.DataFrame(iris.target) 
y.columns = ['Targets'] 

from sklearn.cluster import KMeans 
model = KMeans(n_clusters=3) # clustering data
model.fit(X) 
model.labels_  

import numpy as np 
colormap = np.array(['red', 'lime', 'black'])

import matplotlib.pyplot as plt
plt.figure(figsize=(14,7))   # ploting dataset and comparing Kmeans and EM
plt.subplot(1, 2, 1) 
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40) 
plt.title('EM Clustering') 
plt.subplot(1, 2, 2) 
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40) 
plt.title('K-Means clustering') 

import sklearn.metrics as sm 
acc = sm.accuracy_score( y ,model.labels_) # calculating accuracy
print(acc * 100)