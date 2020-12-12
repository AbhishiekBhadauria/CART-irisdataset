import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
dataset=pd.read_csv("Iris.csv",index_col=False)
print(dataset.head())
dataset['Species']=dataset['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
array=dataset.values
print(array.shape)
X=array[:,1:5]
Y=array[:,5]
print(X.shape)
print(Y.shape)
print(X[0])
print(Y[0])
class_counts=dataset.groupby('Species').size()
print(class_counts)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33)
clf=DecisionTreeClassifier(random_state=10,max_depth=2)
clf.fit(X_train,Y_train)
pred=clf.predict(X_train)
print(accuracy_score(Y_train,pred))
featurename=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
classname=['setosa','versicolor','virginica']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)
tree.plot_tree(clf,feature_names=featurename,class_names=classname,filled=True)
fig.savefig('treevisualization.png')