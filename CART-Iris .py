##Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
#Reading The File
dataset=pd.read_csv("Iris.csv",index_col=False)
#Print Dataset Head
print(dataset.head())
#Mapping Textual Values Into Numerical Values For Machine To Understand It.
dataset['Species']=dataset['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
#Taking Values Of Dataset
array=dataset.values
print(array.shape)
X=array[:,1:5]
Y=array[:,5]
print(X.shape)
print(Y.shape)
print(X[0])
print(Y[0])
#Counting Number Of Instances For Each Class
class_counts=dataset.groupby('Species').size()
print(class_counts)
#Train Test Split The Entire Dataset
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=10)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
#Defining Decision Tree Cassifier
clf=DecisionTreeClassifier(random_state=10,max_depth=3)
#Fitting The Training Data
clf.fit(X_train,Y_train)
#Predicting On Test Dataset
pred=clf.predict(X_test)
#Accuracy Of Y_test And Pred
print(accuracy_score(Y_test,pred))
#Visualizing The Decision Tree
featurename=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
classname=['setosa','versicolor','virginica']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)
tree.plot_tree(clf,feature_names=featurename,class_names=classname,filled=True)
fig.savefig('treevisualization.png')
