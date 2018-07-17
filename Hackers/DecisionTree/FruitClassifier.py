import pandas as pd
import numpy as np
from sklearn import tree #training the decision tree classifier on the loaded dataset

#initialize the dataset
fruit_data_set = pd.DataFrame()
fruit_data_set["fruit"]=np.array([1,1,1,1,1, #1 for apple
                                  0,0,0,0,0]) #0 for orange
fruit_data_set["weight"]=np.array([170,175,180,178,182,
                                   130,120,130,138,145])
fruit_data_set["smooth"]=np.array([9,10,8,8,7,
                                   3,4,2,5,6])

#build the fruit classifier with decision tree algorithm
fruit_classifier=tree.DecisionTreeClassifier() #create the decision tree classifier instance
fruit_classifier.fit(fruit_data_set[["weight","smooth"]],fruit_data_set["fruit"]) #use the loaded fruit dataset features and the target to train the decision tree model

print("Trained fruit_classifier")
print(fruit_classifier)

#fruit dataset 1st observation
test_feature_1=[[fruit_data_set["weight"][0],fruit_data_set["smooth"][0]]]
test_feature_1_fruit=fruit_classifier.predict(test_feature_1)
print("Actual fruite type: {}, Fruit classifier predicted: {}".format(fruit_data_set["fruit"][0],test_feature_1_fruit))


#fruit dataset 3rd observation
test_feature_3=[[fruit_data_set["weight"][2],fruit_data_set["smooth"][2]]]
test_feature_3_fruit=fruit_classifier.predict(test_feature_3)
print("Actual fruite type: {}, Fruit classifier predicted: {}".format(fruit_data_set["fruit"][2],test_feature_3_fruit))


#fruit dataset 8th observation
test_feature_8=[[fruit_data_set["weight"][7],fruit_data_set["smooth"][7]]]
test_feature_8_fruit=fruit_classifier.predict(test_feature_1)
print("Actual fruite type: {}, Fruit classifier predicted: {}".format(fruit_data_set["fruit"][7],test_feature_8_fruit))

#fruit random test 1

#fruit dataset 1st observation
test_feature_10=[[177,9]]
test_feature_10_fruit=fruit_classifier.predict(test_feature_10)
print("Fruit classifier predicted: {}".format(test_feature_10_fruit)) #seems right

with open("Fruit_classifier.txt","w") as f:
    f=tree.export_graphviz(fruit_classifier,out_file=f)


#visulaize the decision tree as pdf
with open("fruit_classifer.dot","w") as f:
    f=tree.export_graphviz(fruit_classifier,out_file=f)