import pandas as pd
from tmdbv3api import TMDb # this is TMDB library
import requests
import ast
import json

train_set = pd.read_csv('../data/train.csv')
# If its taking too long comment out this next line.
tmdb = TMDb()
key = 'e4f05a4f127ed9ce6df860bbdc59d597'
tmdb.api_key = key


def get_popularity(actor_id):
    url = "https://api.themoviedb.org/3/person/" + actor_id + "?api_key=" + key + "&language=en-US"
    response = requests.get(url)
    # print(url + "  ---> " + response.text)
    parsed_json = json.loads(response.text)
    try:
        return parsed_json['popularity']
    except:
        return 0


def get_key_actors():
    actors1 = []
    actors2 = []
    actors3 = []

    big_count = 0
    for index, row in dataset.iterrows():
        is_entered1 = False
        is_entered2 = False
        is_entered3 = False
        try:
            cast = row['cast']
            count = 1
            cast = cast.replace("None", "\"None\"")
            cast_JSON = ast.literal_eval(cast)
            for actor in cast_JSON:
                # print("actor")
                actor_id = actor['id']
                if count == 1:
                    popularity = get_popularity(str(actor_id))
                    actors1.append(popularity)
                    print("count" + str(count))
                    is_entered1 = True
                elif count == 2:
                    popularity = get_popularity(str(actor_id))
                    actors2.append(popularity)
                    print("count" + str(count))
                    is_entered2 = True
                elif count == 3:
                    popularity = get_popularity(str(actor_id))
                    actors3.append(popularity)
                    print("count" + str(count))
                    is_entered3 = True
                else:  # Take the first 3 people
                    break
                # print("######")
                # print(actor['name'], end=' ')
                # print(popularity)
                count = count + 1
            if is_entered1 is False:
                print("False1")
                actors1.append(0.6)
            if is_entered2 is False:
                print("False2")
                actors2.append(0.6)
            if is_entered3 is False:
                print("False3")
                actors3.append(0.6)
        except:  # If any problem occurs with cast variable just say that they all have 0.6 popularity
            print("except")
            if is_entered1 is False:
                print("False1")
                actors1.append(0.6)
            if is_entered2 is False:
                print("False2")
                actors2.append(0.6)
            if is_entered3 is False:
                print("False3")
                actors3.append(0.6)
        print(str(big_count+1), end=' ')
        print(" - " + str(len(actors1)) + " , " + str(len(actors2)) + " , " + str(len(actors3)))
        if big_count+1 != len(actors1) or big_count+1 != len(actors2) or big_count+1 != len(actors3):
            print("ERROR Line 1: Problem occured in, ")
            print("ERROR Line 2: index " + str(index-1))
            print("ERROR Line 3: Please continue preprocessing after that index.")
            return actors1, actors2, actors3
        big_count = big_count + 1
    return actors1, actors2, actors3

"""""
size = 3000
times = 30
proportion = int(size / times)  # make it dividable
for i in range(1, times):
    if i == 1:
        dataset = train_set.loc[0:i*proportion, ]
    else:
        dataset = train_set.loc[((i-1)*proportion)+1:i*proportion, ]
"""
dataset = train_set.loc[0:]
actors1, actors2, actors3 = get_key_actors()
dataset.insert(1, column='actor1', value=actors1)
dataset.insert(2, column='actor2', value=actors2)
dataset.insert(3, column='actor3', value=actors3)
dataset.to_csv("../data/processed_train.csv", sep=',')
print("data_processed is done.")