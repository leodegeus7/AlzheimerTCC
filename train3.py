# coding: utf-8

# In[2]:


# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# load the user configs
with open('conf/conf.json') as f:    
  config = json.load(f)


# In[3]:


test_size     = config["test_size"]
seed      = config["seed"]
features_path   = config["features_path"]
labels_path   = config["labels_path"]
results     = config["results"]
classifier_path = config["classifier_path"]
train_path    = config["train_path"]
num_classes   = config["num_classes"]
classifier_path = config["classifier_path"]
matrix_path = config["matrix_path"]

# In[4]:


# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] training started...")
# split the training and testing data


# In[6]:


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []


# In[ ]:


(trainData2, testData2, trainLabels2, testLabels2) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=0.0,
                                                                  random_state=7)


predsToTest = []
testsToCompare = []

# In[9]:
f = open(results, "w")
count = 1
model2 = LogisticRegression(random_state=seed)
for train, test in kfold.split(trainData2, trainLabels2):
    testLabels = trainLabels2[test]
    testData = trainData2[test]
    trainData = trainData2[train]
    trainLabels = trainLabels2[train]
    model = LogisticRegression(random_state=seed)
    model.fit(trainData, trainLabels)
    
    rank_1 = 0
    rank_5 = 0

    # loop over test data
    for (label, features) in zip(testLabels, testData):
      # predict the probability of each class label and
      # take the top-5 class labels
      predictions = model.predict_proba(np.atleast_2d(features))[0]
      predictions = np.argsort(predictions)[::-1][:5]

      # rank-1 prediction increment
      if label == predictions[0]:
        rank_1 += 1

      # rank-5 prediction increment
      if label in predictions:
        rank_5 += 1

    # convert accuracies to percentages
    rank_1 = (rank_1 / float(len(testLabels))) * 100
    rank_5 = (rank_5 / float(len(testLabels))) * 100
    f.write("Rank-1: {:.2f}%\n".format(rank_1))
    f.write("Rank-5: {:.2f}%\n\n".format(rank_5))
    print("%.3f%%" % (rank_1))
    cvscores.append(rank_1)
    
    preds = model.predict(testData)



    predsToTest = np.concatenate((predsToTest, preds), axis=0)

    testsToCompare = np.concatenate((testsToCompare, testLabels), axis=0)

    f.write("{}\n".format(classification_report(testLabels, preds)))
    print ("[INFO] confusion matrix")
    
    # get the list of training lables
    #labels = sorted(list(os.listdir(train_path)))

    # plot the confusion matrix
    cm = confusion_matrix(testLabels, preds)
 
    sns.heatmap(cm,
                annot=True,
                cmap="Set2")


    nameMatrix = "/".join(matrix_path.split("/")[:-1]) + "/matrix" + str(count) + ".png"
    plt.savefig(nameMatrix)
    plt.close()
    print(nameMatrix)
    count += 1
    model2 = model


for resu in cvscores:
    f.write("\n{:.2f}%\n".format(resu))

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
f.write("Mean: {:.2f}%\n".format(np.mean(cvscores)))


f.write("\nResultados Finais")
f.write("{}\n".format(classification_report(testsToCompare, predsToTest)))

cm = confusion_matrix(testsToCompare, predsToTest)
sns.heatmap(cm,
                annot=True,e
                cmap="Set2")

nameMatrix = "/".join(matrix_path.split("/")[:-1]) + "/matrix" + "Final" + ".png"
plt.savefig(nameMatrix)
plt.close()

f.write("\nAccuracy Folds")
f.write("\n".format(np.std(cvscores)))


f.close()
