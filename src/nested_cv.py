import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train_set = pd.read_csv('../extraData/train_final.csv')

train_set = train_set.reset_index()

train_set = train_set.replace([np.inf, -np.inf], 0)

# Initialise the Scaler
scaler = StandardScaler()

# To scale data
scaler.fit(train_set)

""""
for index, row in train_set.iterrows():
    for item in row:
        if item is not np.float64:
            print("WTF")
"""

print(train_set.shape)

X_data = train_set.drop(['revenue'], axis=1, inplace=False)
y_data = train_set['revenue']


# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}

# We will use MLPRegressor with solver adam
mlp = MLPRegressor(hidden_layer_sizes=(5))
parameters = {'solver': ['adam'], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}

NUM_TRIALS = 30

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

# DOESN'T WORK YET.

# Loop for each trial
for i in range(NUM_TRIALS):
    print(i)
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    inner_cv = KFold(n_splits=2, shuffle=False, random_state=i)
    outer_cv = KFold(n_splits=2, shuffle=False, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=mlp, param_grid=parameters, cv=inner_cv)
    clf.fit(X_data, y_data)
    non_nested_scores[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_data, y=y_data, cv=outer_cv)
    nested_scores[i] = nested_score.mean()

score_difference = non_nested_scores - nested_scores

print("Average difference of {0:6f} with std. dev. of {1:6f}."
      .format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
nested_line, = plt.plot(nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
          x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")

plt.show()
