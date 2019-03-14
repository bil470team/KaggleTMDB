<h3> What is this branch about?

This is the branch README for branch alperen.

Due to that our problem is Regression problem I wanted to use [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor) for my problem.

I tend to use Nested CV for more accurate guesses. But first I will implement MLPRegressor without Nested CV.

<h3> Assumptions

* These features do NOT affect the revenue:
    * runtime
    * poster_path
    * homepage
    * overview
    * tagline
* These features do NOT affect the revenue DIRECTLY. But should be restored for data_checking and more info etc.
    * id
    * imdb_id
    * original_title
    * cast
    * crew
* These features are arrays with multiple valueable data in it. So it is essential to figure out how to fit in the train algorithm:
    * keywords
    * genres

* If 'status' is RUMORED then the revenue is definitely 0. Yet, in all train data there is no such instance.   

<h4> TODO

1. Decide if title or original_title matters.
1. How to use belongs_to_collection
1. Find out a way to train data with key actors and such.
1. Find out how keywords and genres are going to be used.
1. ...