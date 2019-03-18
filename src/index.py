# This is the main python file for machine learning models.
# Every change in src/ folder is accepted.
# Yet don't disturb the data/ folder
import pandas as pd
print("Hey there Developer")
dataset = pd.read_csv("../data/train.csv")
y = dataset['revenue']
X = dataset.drop(['revenue'], axis = 1)