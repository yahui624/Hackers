from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import graphviz

iris=load_iris()

#create decision tree classifier and train on testing date
df=pd.DataFrame(iris.data,columns=iris.feature_names)
dtree=DecisionTreeClassifier()
dtree.fit(df,iris.target)

#visualize the tree
dot_data=StringIO()

with open("iris_classifier.txt","w") as f:
    f=export_graphviz(dtree,out_file=f)

with open("iris_classifer.dot","w") as f:
    f=export_graphviz(dtree,out_file=f,filled=True,feature_names=iris.feature_names, class_names=iris.target_names,rounded=True,impurity=False,special_characters=True)