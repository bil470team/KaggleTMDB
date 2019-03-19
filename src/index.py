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


# for example if the name is homepage it creates a column as has_homepage.
def create_column(name):
    has_name = 'has_'+name
    X_train[has_name] = 0
    X_train.loc[X_train[name].isnull() == False, has_name] = 1
    X_test[has_name] = 0
    X_test.loc[X_test[name].isnull() == False, has_name] = 1
    # Now because the column has_status is created "status" is no longer needed, so drop it.
    X_train.drop([name], axis=1, inplace=True)
    X_test.drop([name], axis=1, inplace=True)


create_column('homepage')
create_column('overview')
create_column('tagline')
create_column('status')

sns.set(style = "ticks")
sns.barplot(x="popularity", y="revenue", data=dataset)
