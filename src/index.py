# This is the main python file for machine learning models.
# Every change in src/ folder is accepted.
# Yet don't disturb the data/ folder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
# Creating Data and Value DataSets
dataset = pd.read_csv("../data/train.csv")
testset= pd.read_csv("../data/test.csv")
#Dropping imdb_id and poster_path features as they are irrelevant.
#Also dropping id because it is just an index for dataset.
dataset.drop(['imdb_id','poster_path', 'id'],axis=1,inplace=True)
testset.drop(['imdb_id','poster_path', 'id'],axis=1,inplace=True)
y_train = dataset['revenue']
#dataset.drop(['revenue'], axis = 1, inplace = True)
X_train = dataset
X_test = testset
#PROCESSING HOMEPAGE
#Create a new column "has_homepage" which will be 1 if the movie has a homepage and 0 if it does not.
X_train['has_homepage'] = 0
X_train.loc[X_train['homepage'].isnull() == False, 'has_homepage'] = 1
X_test['has_homepage'] = 0
X_test.loc[X_test['homepage'].isnull() == False, 'has_homepage'] = 1
#Now because the column has_homepage is created "homepage" is no longer needed, so drop it.
X_train.drop(['homepage'], axis = 1, inplace = True)
X_test.drop(['homepage'], axis = 1, inplace = True)

#PROCESSING OVERVIEW
#Create a new column "has_overview" which will be 1 if the movie has an overview and 0 if it does not.
X_train['has_overview'] = 0
X_train.loc[X_train['overview'].isnull() == False, 'has_overview'] = 1
X_test['has_overview'] = 0
X_test.loc[X_test['overview'].isnull() == False, 'has_overview'] = 1
#Now because the column has_overview is created "overview" is no longer needed, so drop it.
X_train.drop(['overview'], axis = 1, inplace = True)
X_test.drop(['overview'], axis = 1, inplace = True)

#PROCESSING TAGLINE
#Create a new column "has_tagline" which will be 1 if the movie has a tagline and 0 if it does not.
X_train['has_tagline'] = 0
X_train.loc[X_train['tagline'].isnull() == False, 'has_tagline'] = 1
X_test['has_tagline'] = 0
X_test.loc[X_test['tagline'].isnull() == False, 'has_tagline'] = 1
#Now because the column has_tagline is created "tagline" is no longer needed, so drop it.
X_train.drop(['tagline'], axis = 1, inplace = True)
X_test.drop(['tagline'], axis = 1, inplace = True)

#PROCESSING STATUS
#Create a new column "has_tagline" which will be 1 if the movie has a tagline and 0 if it does not.
X_train['has_status'] = 0
X_train.loc[X_train['status'].isnull() == False, 'has_status'] = 1
X_test['has_status'] = 0
X_test.loc[X_test['status'].isnull() == False, 'has_status'] = 1
#Now because the column has_status is created "status" is no longer needed, so drop it.
X_train.drop(['status'], axis = 1, inplace = True)
X_test.drop(['status'], axis = 1, inplace = True)

sns.set(style = "ticks")
sns.barplot(x="popularity", y="revenue", data=dataset)
