#! python 3
# msd_imp.py - recommendation system for million song challenge using implicit

## Import necessary libraries
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import metrics
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as ms
from surprise import SVD
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise import Dataset
from surprise.model_selection import KFold
from surprise import SVDpp
from surprise.model_selection import GridSearchCV






# Training data
train_trip = pd.read_csv('train_triplets.txt', delimiter = '\t', header = None)
train_trip.columns = ['users', 'songs', 'play count']

# Test data
test_trip = pd.read_csv('kaggle_visible_evaluation_triplets.txt', delimiter = '\t', header = None, names = ['users', 'songs', 'play count'])



## Surprise package 



# Create reader object for defining rating scale

reader = Reader(rating_scale = (0,max(train_trip['users'])))

# Convert the dataframe into surprise-friendly object with ratings scale

surp_df = Dataset.load_from_df(train_trip, reader)

# Grid search object

param_grid = {'n_epochs': [5,10,15], 'lr_all': [0.002, 0.005, 0.007],
              'reg_all': [0.2, 0.4, 0.6]}

gs = GridSearchCV(SVDpp, param_grid, measures = ['rmse', 'mae'], cv = 3,
                  verbose = True)

# Fit the data

gs.fit(surp_df)

# Determine best parameters for both metrics

print(gs.cv_results_)

print(gs.best_score_['rmse'])

print(gs.best_params['mae'])


algo = gs.best_estimator['rmse']

algo.fit(train_trip)

# Fit on the test data

preds = algo.test(test_trip)

# compute accuracy utilizing the rmse

accuracy.rmse(predictions, verbose = True)