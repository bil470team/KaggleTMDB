import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

train_set = pd.read_csv('../extraData/train_final.csv')

train_set = train_set.reset_index()

train_set = train_set.replace([np.inf, -np.inf], 0)

# Initialise the Scaler
scaler = StandardScaler()

# To scale data
scaler.fit_transform(train_set)

train_set.to_csv("trial.csv")

y_train = train_set['revenue']
X_train = train_set.drop(['revenue'], axis=1)

test1 = pd.read_csv('../extraData/test_final.csv')
test2 = pd.read_csv('../extraData/test_final.csv')

test1 = test1.replace([np.inf, -np.inf], 0)
test2 = test2.replace([np.inf, -np.inf], 0)


def test_partial_fit_regression(X, y):
    # Test partial_fit on regression.
    # `partial_fit` should yield the same results as 'fit' for regression.

    predictions1 = []
    predictions2 = []
    score = []

    for momentum in [0, .9]:
        mlp = MLPRegressor(solver='adam', max_iter=1000, activation='relu',
                           random_state=1, learning_rate_init=0.01,
                           batch_size=X.shape[0], momentum=momentum)
        with warnings.catch_warnings(record=True):
            # catch convergence warning
            mlp.fit(X, y)
        predictions1 = mlp.predict(test1)
        mlp = MLPRegressor(solver='adam', activation='relu',
                           learning_rate_init=0.01, random_state=1,
                           batch_size=X.shape[0], momentum=momentum)
        for i in range(100):
            mlp.partial_fit(X, y)

        predictions2 = mlp.predict(test2)
        # npt.assert_almost_equal(pred1, pred2, decimal=2)
        score = mlp.score(X, y)

    print(len(predictions1))
    test1['revenue'] = predictions1
    test1[['id', 'revenue']].to_csv('../submissions/submission1.csv', index=False)

    test2['revenue'] = predictions2
    test2[['id', 'revenue']].to_csv('../submissions/submission2.csv', index=False)


def gridSearchCV_with_MLP(X, y):
    test3 = pd.read_csv('../extraData/test_final.csv')
    test3 = test3.replace([np.inf, -np.inf], 0)

    parameters = {'solver': ['adam'],
                  'max_iter': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
                  'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes': np.arange(10, 15)}

    clf = MLPRegressor()

    grid_obj = GridSearchCV(clf, parameters, scoring=None, cv=3, n_jobs=-1)
    grid_obj = grid_obj.fit(X, y)
    clf = grid_obj.best_estimator_
    clf.fit(X, y)

    print(clf.score(X, y))
    print(clf.best_params_)

    predictions3 = clf.predict(test3)

    test3['revenue'] = predictions3
    test3[['id', 'revenue']].to_csv('../submissions/submission3.csv', index=False)


# Partial and normal MLP
# test_partial_fit_regression(X_train, y_train)

# GridSearchCV with MLP.
gridSearchCV_with_MLP(X_train, y_train)
