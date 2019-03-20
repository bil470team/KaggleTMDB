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
dataset.drop(['imdb_id','poster_path', 'original_title'],axis=1,inplace=True)
testset.drop(['imdb_id','poster_path', 'original_title'],axis=1,inplace=True)
y_train = dataset['revenue']
#dataset.drop(['revenue'], axis = 1, inplace = True)
X_train = dataset
X_test = testset
#testset.drop(['revenue'], axis = 1, inplace = True)



# for example if the name is homepage it creates a column as has_homepage.
def create_column(name):
    has_name = 'has_'+name
    X_train[has_name] = 0
    X_train.loc[X_train[name].isnull() == False, has_name] = 1
    X_test[has_name] = 0
    X_test.loc[X_test[name].isnull() == False, has_name] = 1
    # Now because the column has_homepage is created "homepage" is no longer needed, so drop it.
    X_train.drop([name], axis=1, inplace=True)
    X_test.drop([name], axis=1, inplace=True)

create_column('homepage')
create_column('overview')
create_column('tagline')
create_column('status')
create_column('belongs_to_collection')

def create_json_column(columnNameToProcess, fieldName, dataFrame):
    name = []
    for i in dataFrame[columnNameToProcess]:
        if(not(pd.isnull(i))):
            if (eval(i)[0]['name'] == fieldName):
                name.append(1)
            else:
                name.append(0)
        else:
            name.append(0)
    return name

genres_list = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
               'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 
               'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
               'Thriller', 'War', 'Western']

for i in genres_list:
    nameForColumn = 'genre_'+i
    X_train[nameForColumn] = create_json_column('genres', i, X_train)
    X_test[nameForColumn] = create_json_column('genres', i, X_test)
    
X_train.drop(['genres'], axis=1, inplace=True)
X_test.drop(['genres'], axis=1, inplace=True)


for c in ['original_language', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords']:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(X_train[c].fillna('').astype(str)) + list(X_test[c].fillna('').astype(str)))
    X_train[c] = lbl.transform(X_train[c].fillna(''))
    X_test[c] = lbl.transform(X_test[c].fillna(''))
    
