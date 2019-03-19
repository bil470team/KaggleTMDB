import pandas as pd
import numpy as np
from tmdbv3api import TMDb # this is TMDB library
import requests
import ast
import json

dataset = pd.read_csv('../data/train.csv')
tmdb = TMDb()
key = 'e4f05a4f127ed9ce6df860bbdc59d597'
tmdb.api_key = key


def get_popularity(actor_id):
    url = "https://api.themoviedb.org/3/person/" + actor_id + "?api_key=" + key + "&language=en-US"
    response = requests.get(url)
    print(url + "  ---> " + response.text)
    parsed_json = json.loads(response.text)
    try:
        return parsed_json['popularity']
    except:
        return 0


def get_key_actors():
    actors1 = actors2 = actors3 = actors4 = actors5 = []
    for index, row in dataset.iterrows():
        cast = row['cast']
        count = 1
        #cast = cast.replace('\'', '\"')
        cast = cast.replace("None", "\"None\"")
        cast_JSON = ast.literal_eval(cast)
        for actor in cast_JSON:
            try:
                actor_id = actor['id']
                if count == 1:
                    popularity = get_popularity(str(actor_id))
                    actors1.append(popularity)
                elif count == 2:
                    popularity = get_popularity(str(actor_id))
                    actors2.append(popularity)
                elif count == 3:
                    popularity = get_popularity(str(actor_id))
                    actors3.append(popularity)
                elif count == 4:
                    popularity = get_popularity(str(actor_id))
                    actors4.append(popularity)
                elif count == 5:
                    popularity = get_popularity(str(actor_id))
                    actors5.append(popularity)
                else:  # Take the first 5 people
                    break
            except:  # If the cast is smaller than 5 people
                break
            print("######")
            print(actor['name'], end=' ')
            print(popularity)
            count = count + 1
    return actors1, actors2, actors3, actors4, actors5

get_key_actors()