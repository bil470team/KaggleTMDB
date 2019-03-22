import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
import catboost as cat

from tqdm import tqdm
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from collections import Counter
from sklearn.model_selection import GroupKFold

import warnings
warnings.filterwarnings("ignore")

random_seed = 42
le = LabelEncoder()

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train.index = train['id']
test.index = test['id']

#An extra training data.
release_dates = pd.read_csv('../extraData/release_dates_per_country.csv')

#Data scraping done by us.
#Related code is located at: https://github.com/bil470team/KaggleTMDB/blob/preprocessing/src/alperen_index.py
actors = pd.read_csv('../extraData/processed_test.csv')[['imdb_id', 'actor1', 'actor2', 'actor3']]

release_dates['id'] = range(1,7399)
release_dates.drop(['original_title','title'],axis = 1,inplace = True)
release_dates.index = release_dates['id']

train = pd.merge(train, release_dates, how='left', on=['id'])
test = pd.merge(test, release_dates, how='left', on=['id'])

#Extra tranining data, acquired from Kaggle public datasets.
trainAdditionalFeatures = pd.read_csv('../extraData/TrainAdditionalFeatures.csv')[['imdb_id','popularity2','rating', 'totalVotes']]
testAdditionalFeatures = pd.read_csv('../extraData/TestAdditionalFeatures.csv')[['imdb_id','popularity2','rating', 'totalVotes']]

#Filling NA's with mean did not benefit, so dropping them.
trainAdditionalFeatures = trainAdditionalFeatures.dropna()
testAdditionalFeatures = testAdditionalFeatures.dropna()

train = pd.merge(train, trainAdditionalFeatures, how='left', on=['imdb_id'])
train = pd.merge(train, actors, how='left', on=['imdb_id'])

test = pd.merge(test, testAdditionalFeatures, how='left', on=['imdb_id'])
test = pd.merge(test, actors, how='left', on=['imdb_id'])

#Extra data included some missing data, it is best to fill them with mean.
train['rating'].fillna(train['rating'].mean(), inplace = True)
train['popularity2'].fillna(train['popularity2'].mean(), inplace = True)
train['totalVotes'].fillna(train['totalVotes'].mean(), inplace = True)

test['rating'].fillna(test['rating'].mean(), inplace= True)
test['popularity2'].fillna(test['popularity2'].mean(), inplace = True)
test['totalVotes'].fillna(test['totalVotes'].mean(), inplace = True)


#Handling missing or very-erroneous datas. Acquired from Kaggle.
def clean_train_data():
    train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning
    train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby
    train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs
    train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven
    train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout
    train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty
    train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood
    train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II
    train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada
    train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol
    train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
    train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times
    train.loc[train['id'] == 1007,'budget'] = 2              # Zyzzyx Road
    train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman
    train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit
    train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy
    train.loc[train['id'] == 1542,'budget'] = 1              # All at Once
    train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II
    train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp
    train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit
    train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon
    train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed
    train.loc[train['id'] == 1885,'budget'] = 12             # In the Cut
    train.loc[train['id'] == 2091,'budget'] = 10             # Deadfall
    train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget
    train.loc[train['id'] == 2491,'budget'] = 6              # Never Talk to Strangers
    train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus
    train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams
    train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D
    train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture
    train.loc[train['id'] == 335,'budget'] = 2
    train.loc[train['id'] == 348,'budget'] = 12
    train.loc[train['id'] == 470,'budget'] = 13000000
    train.loc[train['id'] == 513,'budget'] = 1100000
    train.loc[train['id'] == 640,'budget'] = 6
    train.loc[train['id'] == 696,'budget'] = 1
    train.loc[train['id'] == 797,'budget'] = 8000000
    train.loc[train['id'] == 850,'budget'] = 1500000
    train.loc[train['id'] == 1199,'budget'] = 5
    train.loc[train['id'] == 1282,'budget'] = 9               # Death at a Funeral
    train.loc[train['id'] == 1347,'budget'] = 1
    train.loc[train['id'] == 1755,'budget'] = 2
    train.loc[train['id'] == 1801,'budget'] = 5
    train.loc[train['id'] == 1918,'budget'] = 592
    train.loc[train['id'] == 2033,'budget'] = 4
    train.loc[train['id'] == 2118,'budget'] = 344
    train.loc[train['id'] == 2252,'budget'] = 130
    train.loc[train['id'] == 2256,'budget'] = 1
    train.loc[train['id'] == 2696,'budget'] = 10000000
clean_train_data()

def clean_test_data():
    test.loc[test['id'] == 6733,'budget'] = 5000000
    test.loc[test['id'] == 3889,'budget'] = 15000000
    test.loc[test['id'] == 6683,'budget'] = 50000000
    test.loc[test['id'] == 5704,'budget'] = 4300000
    test.loc[test['id'] == 6109,'budget'] = 281756
    test.loc[test['id'] == 7242,'budget'] = 10000000
    test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family
    test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage
    test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee
    test.loc[test['id'] == 3033,'budget'] = 250
    test.loc[test['id'] == 3051,'budget'] = 50
    test.loc[test['id'] == 3084,'budget'] = 337
    test.loc[test['id'] == 3224,'budget'] = 4
    test.loc[test['id'] == 3594,'budget'] = 25
    test.loc[test['id'] == 3619,'budget'] = 500
    test.loc[test['id'] == 3831,'budget'] = 3
    test.loc[test['id'] == 3935,'budget'] = 500
    test.loc[test['id'] == 4049,'budget'] = 995946
    test.loc[test['id'] == 4424,'budget'] = 3
    test.loc[test['id'] == 4460,'budget'] = 8
    test.loc[test['id'] == 4555,'budget'] = 1200000
    test.loc[test['id'] == 4624,'budget'] = 30
    test.loc[test['id'] == 4645,'budget'] = 500
    test.loc[test['id'] == 4709,'budget'] = 450
    test.loc[test['id'] == 4839,'budget'] = 7
    test.loc[test['id'] == 3125,'budget'] = 25
    test.loc[test['id'] == 3142,'budget'] = 1
    test.loc[test['id'] == 3201,'budget'] = 450
    test.loc[test['id'] == 3222,'budget'] = 6
    test.loc[test['id'] == 3545,'budget'] = 38
    test.loc[test['id'] == 3670,'budget'] = 18
    test.loc[test['id'] == 3792,'budget'] = 19
    test.loc[test['id'] == 3881,'budget'] = 7
    test.loc[test['id'] == 3969,'budget'] = 400
    test.loc[test['id'] == 4196,'budget'] = 6
    test.loc[test['id'] == 4221,'budget'] = 11
    test.loc[test['id'] == 4222,'budget'] = 500
    test.loc[test['id'] == 4285,'budget'] = 11
    test.loc[test['id'] == 4319,'budget'] = 1
    test.loc[test['id'] == 4639,'budget'] = 10
    test.loc[test['id'] == 4719,'budget'] = 45
    test.loc[test['id'] == 4822,'budget'] = 22
    test.loc[test['id'] == 4829,'budget'] = 20
    test.loc[test['id'] == 4969,'budget'] = 20
    test.loc[test['id'] == 5021,'budget'] = 40
    test.loc[test['id'] == 5035,'budget'] = 1
    test.loc[test['id'] == 5063,'budget'] = 14
    test.loc[test['id'] == 5119,'budget'] = 2
    test.loc[test['id'] == 5214,'budget'] = 30
    test.loc[test['id'] == 5221,'budget'] = 50
    test.loc[test['id'] == 4903,'budget'] = 15
    test.loc[test['id'] == 4983,'budget'] = 3
    test.loc[test['id'] == 5102,'budget'] = 28
    test.loc[test['id'] == 5217,'budget'] = 75
    test.loc[test['id'] == 5224,'budget'] = 3
    test.loc[test['id'] == 5469,'budget'] = 20
    test.loc[test['id'] == 5840,'budget'] = 1
    test.loc[test['id'] == 5960,'budget'] = 30
    test.loc[test['id'] == 6506,'budget'] = 11
    test.loc[test['id'] == 6553,'budget'] = 280
    test.loc[test['id'] == 6561,'budget'] = 7
    test.loc[test['id'] == 6582,'budget'] = 218
    test.loc[test['id'] == 6638,'budget'] = 5
    test.loc[test['id'] == 6749,'budget'] = 8
    test.loc[test['id'] == 6759,'budget'] = 50
    test.loc[test['id'] == 6856,'budget'] = 10
    test.loc[test['id'] == 6858,'budget'] = 100
    test.loc[test['id'] == 6876,'budget'] = 250
    test.loc[test['id'] == 6972,'budget'] = 1
    test.loc[test['id'] == 7079,'budget'] = 8000000
    test.loc[test['id'] == 7150,'budget'] = 118
    test.loc[test['id'] == 6506,'budget'] = 118
    test.loc[test['id'] == 7225,'budget'] = 6
    test.loc[test['id'] == 7231,'budget'] = 85
    test.loc[test['id'] == 5222,'budget'] = 5
    test.loc[test['id'] == 5322,'budget'] = 90
    test.loc[test['id'] == 5350,'budget'] = 70
    test.loc[test['id'] == 5378,'budget'] = 10
    test.loc[test['id'] == 5545,'budget'] = 80
    test.loc[test['id'] == 5810,'budget'] = 8
    test.loc[test['id'] == 5926,'budget'] = 300
    test.loc[test['id'] == 5927,'budget'] = 4
    test.loc[test['id'] == 5986,'budget'] = 1
    test.loc[test['id'] == 6053,'budget'] = 20
    test.loc[test['id'] == 6104,'budget'] = 1
    test.loc[test['id'] == 6130,'budget'] = 30
    test.loc[test['id'] == 6301,'budget'] = 150
    test.loc[test['id'] == 6276,'budget'] = 100
    test.loc[test['id'] == 6473,'budget'] = 100
    test.loc[test['id'] == 6842,'budget'] = 30
clean_test_data()

#Handling release_date
def handle_release_date(df):
    df[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[ (df['release_year'] <= 18) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[ (df['release_year'] > 18)  & (df['release_year'] < 100), "release_year"] += 1900
    releaseDate = pd.to_datetime(df['release_date'])
    df['release_dayofweek'] = releaseDate.dt.dayofweek #The day of release affects the revenue.
    df['release_quarter'] = releaseDate.dt.quarter
handle_release_date(train)
handle_release_date(test)

#Budget in 90's and nowadays can differ hugely because of inflation. So, adjust it.
def handle_inflation():
    train['inflationBudget'] = train['budget'] + train['budget']*1.8/100*(2018-train['release_year']) #Inflation simple formula
    train['budget'] = np.log1p(train['budget'])

    test['inflationBudget'] = test['budget'] + test['budget']*1.8/100*(2018-test['release_year']) #Inflation simple formula
    test['budget'] = np.log1p(test['budget'])
handle_inflation()

#Homepage, collection, tagline and released either true or false. So assign them 1 and 0 respectively.
train['has_homepage'] = 1
train.loc[pd.isnull(train['homepage']), "has_homepage"] = 0

train['has_collection'] = 1
train.loc[pd.isnull(train['belongs_to_collection']), "has_collection"] = 0

train['has_tagline'] = 1
train.loc[pd.isnull(train['tagline']), "has_tagline"] = 0

train['isMovieReleased'] = 0
train.loc[train['status'] == "Released", "isMovieReleased"] = 1

#Some movies have different titles in the local language. Their revenue may had increased due to their nation-wise popularity.
train['isTitleDifferent'] = 0
train.loc[train['original_title'] != train['title'] ,"isTitleDifferent"] = 1

#Create new features for most common & high-grossing languages
train['isOriginalLanguageEng'] = 0
train.loc[train['original_language'] == "en" ,"isOriginalLanguageEng"] = 1

#Calculate ratios which are highly-correlated with revenue.
def calculate_ratios(df):
    df['budget/runtime'] = df['budget']/df['runtime']
    df['budget/popularity'] = df['budget']/df['popularity']
    df['budget/year'] = df['budget']/(df['release_year']*df['release_year'])
    df['budget/rating'] = df['budget']/df['rating']
    df['budget/totalVotes'] = df['budget']/df['totalVotes']
    df['budget/theatrical'] = df['budget']/df['theatrical']
        
    df['rating/popularity'] = df['rating']/df['popularity']

    df['popularity/theatrical'] = df['theatrical']/df['popularity']
    df['popularity/releaseYear'] = df['popularity']/df['release_year']
    df['popularity/totalVotes'] = df['totalVotes']/df['popularity']

    df['releaseYear/popularity'] = df['release_year']/df['popularity']

    df['runtime/rating'] = df['runtime']/df['rating']

    df['totalVotes/releaseYear'] = df['totalVotes']/df['release_year']
calculate_ratios(train)
calculate_ratios(test)


#Means are similar to ratios. Constructs a baseline for predictions.
def calculate_means(df):
    df['meanruntimeByYear'] = df[['release_year']].merge(df.groupby('release_year', as_index=False)['runtime'].agg(np.nanmean), on='release_year', how='left')['runtime']
    df['meanpopularityByYear'] = df[['release_year']].merge(df.groupby('release_year', as_index=False)['popularity'].agg(np.nanmean), on='release_year', how='left')['popularity']
    df['meanbudgetByYear'] = df[['release_year']].merge(df.groupby('release_year', as_index=False)['budget'].agg(np.nanmean), on='release_year', how='left')['budget']
    df['meantotalVotesByYear'] = df[['release_year']].merge(df.groupby('release_year', as_index=False)['totalVotes'].agg(np.nanmean), on='release_year', how='left')['totalVotes']
    df['meantotalVotesByRating'] = df[['rating']].merge(df.groupby('rating', as_index=False)['totalVotes'].agg(np.nanmean), on='rating', how='left')['totalVotes']
    df['medianbudgetByYear'] = df[['release_year']].merge(df.groupby('release_year', as_index=False)['budget'].agg(np.nanmedian), on='release_year', how='left')['budget']
calculate_means(train)
calculate_means(test)


def prepare(df):
    global json_cols
    global train_dict

    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))


    df['_collection_name'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
    le.fit(list(df['_collection_name'].fillna('')))
    df['_collection_name'] = le.transform(df['_collection_name'].fillna('').astype(str))
    df['#Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)
    # get collection id
    df['collection_id'] = df['belongs_to_collection'].apply(lambda x : np.nan if len(x)==0 else x[0]['id'])

    df['original_title_letter_count'] = df['original_title'].str.len()
    df['original_title_word_count'] = df['original_title'].str.split().str.len()


    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()

    df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))
    df['cast_count'] = df['cast'].apply(lambda x : len(x))
    df['crew_count'] = df['crew'].apply(lambda x : len(x))




    for col in ['genres', 'production_countries', 'spoken_languages', 'production_companies'] :
        df[col] = df[col].map(lambda x: sorted(list(set([n if n in train_dict[col] else col+'_etc' for n in [d['name'] for d in x]])))).map(lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1, sort=False)
    df.drop(['genres_etc'], axis=1, inplace=True)

    df = df.drop(['belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'overview', 'runtime',
    'poster_path','production_companies', 'production_countries', 'release_date', 'spoken_languages', 'status', 'title', 'cast', 'crew', 'original_language', 'original_title', 'tagline', 'collection_id','movie_id'
    ],axis=1)

    df.fillna(value=0.0, inplace = True)

    return df

test['revenue'] = np.nan

json_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']


def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


for col in tqdm(json_cols + ['belongs_to_collection']) :
    train[col] = train[col].apply(lambda x : get_dictionary(x))
    test[col] = test[col].apply(lambda x : get_dictionary(x))


def get_json_dict(df):
    global json_cols
    result = dict()
    for e_col in json_cols:
        d = dict()
        rows = df[e_col].values
        for row in rows:
            if row is None:
                continue
            for i in row:
                if i['name'] not in d :
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d
    return result


train_dict = get_json_dict(train)
test_dict = get_json_dict(test)

# remove cateogry with bias and low frequency
for col in json_cols:

    remove = []
    train_id = set(list(train_dict[col].keys()))
    test_id = set(list(test_dict[col].keys()))

    remove += list(train_id - test_id) + list(test_id - train_id)
    for i in train_id.union(test_id) - set(remove):
        if train_dict[col][i] < 10 or i == '':
            remove += [i]

    for i in remove:
        if i in train_dict[col] :
            del train_dict[col][i]
        if i in test_dict[col] :
            del test_dict[col][i]

    print(col, 'size :', len(train_id.union(test_id)), '->', len(train_dict[col]))

all_data = prepare(pd.concat([train, test]).reset_index(drop=True))
train = all_data.loc[:train.shape[0] - 1,:]
test = all_data.loc[train.shape[0]:,:]

#change5
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)
train['all_Keywords'] = train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]
for g in top_keywords:
    train['keyword_' + g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)

test['num_Keywords'] = test['Keywords'].apply(lambda x: len(x) if x != {} else 0)
test['all_Keywords'] = test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_keywords:
    test['keyword_' + g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['Keywords', 'all_Keywords'], axis=1)
test = test.drop(['Keywords', 'all_Keywords'], axis=1)

features = list(train.columns)
features = [i for i in features if i != 'id' and i != 'revenue']


#
#
# ******************** END OF PREPROCESSING
#
#



def score(data, y):
    validation_res = pd.DataFrame(
    {"id": data["id"].values,
     "transactionrevenue": data["revenue"].values,
     "predictedrevenue": np.expm1(y)})

    validation_res = validation_res.groupby("id")["transactionrevenue", "predictedrevenue"].sum().reset_index()
    return np.sqrt(mean_squared_error(np.log1p(validation_res["transactionrevenue"].values),
                                     np.log1p(validation_res["predictedrevenue"].values)))


class KFoldValidation():
    def __init__(self, data, n_splits=5):
        unique_vis = np.array(sorted(data['id'].astype(str).unique()))
        folds = GroupKFold(n_splits)
        ids = np.arange(data.shape[0])

        self.fold_ids = []
        for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
            self.fold_ids.append([
                    ids[data['id'].astype(str).isin(unique_vis[trn_vis])],
                    ids[data['id'].astype(str).isin(unique_vis[val_vis])]
                ])

    def validate(self, train, test, features, model, name="", prepare_stacking=False,
                 fit_params={"early_stopping_rounds": 500, "verbose": 100, "eval_metric": "rmse"}):
        model.FI = pd.DataFrame(index=features)
        full_score = 0

        if prepare_stacking:
            test[name] = 0
            train[name] = np.NaN

        for fold_id, (trn, val) in enumerate(self.fold_ids):
            devel = train[features].iloc[trn]
            y_devel = np.log1p(train["revenue"].iloc[trn])
            valid = train[features].iloc[val]
            y_valid = np.log1p(train["revenue"].iloc[val])

            print("Fold ", fold_id, ":")
            model.fit(devel, y_devel, eval_set=[(valid, y_valid)], **fit_params)

            if len(model.feature_importances_) == len(features):
                model.FI['fold' + str(fold_id)] = model.feature_importances_ / model.feature_importances_.sum()

            predictions = model.predict(valid)
            predictions[predictions < 0] = 0
            print("Fold ", fold_id, " error: ", mean_squared_error(y_valid, predictions)**0.5)

            fold_score = score(train.iloc[val], predictions)
            full_score += fold_score / len(self.fold_ids)
            print("Fold ", fold_id, " score: ", fold_score)
            if prepare_stacking:
                train[name].iloc[val] = predictions

                test_predictions = model.predict(test[features])
                test_predictions[test_predictions < 0] = 0
                test[name] += test_predictions / len(self.fold_ids)

        print("Final score: ", full_score)
        return full_score


Kfolder = KFoldValidation(train)

lgbmodel = lgb.LGBMRegressor(n_estimators=10000,
                             objective='regression',
                             metric='rmse',
                             max_depth = 5,
                             num_leaves=30,
                             min_child_samples=100,
                             learning_rate=0.01,
                             boosting = 'gbdt',
                             min_data_in_leaf= 10,
                             feature_fraction = 0.9,
                             bagging_freq = 1,
                             bagging_fraction = 0.9,
                             importance_type='gain',
                             lambda_l1 = 0.2,
                             bagging_seed=random_seed,
                             subsample=.8,
                             colsample_bytree=.9,
                             use_best_model=True)

Kfolder.validate(train, test, features , lgbmodel, name="lgbfinal", prepare_stacking=True)

xgbmodel = xgb.XGBRegressor(max_depth=5,
                            learning_rate=0.01,
                            n_estimators=10000,
                            objective='reg:linear',
                            gamma=1.45,
                            seed=random_seed,
                            silent=True,
                            subsample=0.8,
                            colsample_bytree=0.7,
                            colsample_bylevel=0.5)

Kfolder.validate(train, test, features, xgbmodel, name="xgbfinal", prepare_stacking=True)

catmodel = cat.CatBoostRegressor(iterations=10000,
                                 learning_rate=0.008,
                                 depth=5,
                                 eval_metric='RMSE',
                                 bagging_temperature = 0.2,
                                 metric_period = None,
                                 early_stopping_rounds=200,
                                 rsm = 0.1,
                                 random_seed=random_seed)

Kfolder.validate(train, test, features , catmodel, name="catfinal", prepare_stacking=True,
               fit_params={"use_best_model": True, "verbose": 100})

train['Revenue_lgb'] = train["lgbfinal"]

print("RMSE model lgb :" ,score(train, train.Revenue_lgb),)

train['Revenue_xgb'] = train["xgbfinal"]

print("RMSE model xgb :" ,score(train, train.Revenue_xgb))

train['Revenue_cat'] = train["catfinal"]

print("RMSE model cat :" ,score(train, train.Revenue_cat))

train['Revenue_Dragon1'] = 0.4 * train["lgbfinal"] + \
                               0.2 * train["xgbfinal"] + \
                               0.4 * train["catfinal"]

print("RMSE model Dragon1 :" ,score(train, train.Revenue_Dragon1))

train['Revenue_Dragon2'] = 0.35 * train["lgbfinal"] + \
                               0.3 * train["xgbfinal"] + \
                               0.35 * train["catfinal"]

print("RMSE model Dragon2 :" ,score(train, train.Revenue_Dragon2))

test['revenue'] =  np.expm1(0.4 * test["lgbfinal"]+ 0.4 * test["catfinal"] + 0.2 * test["xgbfinal"])
test[['id','revenue']].to_csv('submission_Dragon1.csv', index=False)
test[['id','revenue']].head()

test['revenue'] =  np.expm1(test["xgbfinal"])
test[['id','revenue']].to_csv('submission_xgb.csv', index=False)
test[['id','revenue']].head()

test['revenue'] =  np.expm1(test["catfinal"])
test[['id','revenue']].to_csv('submission_cat.csv', index=False)
test[['id','revenue']].head()

test['revenue'] =  np.expm1((test["lgbfinal"] + test["catfinal"] + test["xgbfinal"])/3)
test[['id','revenue']].to_csv('submission_Dragon2.csv', index=False)
test[['id','revenue']].head()
