#! python 3
# msd_imp.py - recommendation system for million song challenge using implicit

## Import necessary libraries
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as ms
import implicit





# Training data
train_trip = pd.read_csv('train_triplets.txt', delimiter = '\t', header = None)
train_trip.columns = ['users', 'songs', 'play count']

# Test data
test_trip = pd.read_csv('kaggle_visible_evaluation_triplets.txt', delimiter = '\t', header = None, names = ['users', 'songs', 'play count'])

# Combine the two dataframes, as this is an implicit dataset

df = pd.concat([train_trip, test_trip], ignore_index = True)

# Make the userid and songid strings
df['users'] = df['users'].astype(str)
df['songs'] = df['songs'].astype(str)

# Inspect data

train_trip.head()
train_trip.shape

# How many unique users

print(len(df['users'].drop_duplicates()))

# How many unique songs

print(len(df['songs'].drop_duplicates()))

# Check for null values

print(df['play count'].isnull().sum())

# Check the max and min of each column

play_count_max = df['play count'].max()
play_count_min = df['play count'].min()


# Check the most popular songids

song_pop = df.songs.value_counts()


# ALS Collaborative Filter using implicit library

# Make user and song into coded factors

df['songs'] = df['songs'].astype("category")
df['users'] = df['users'].astype("category")
df['user_id'] = df['users'].cat.codes
df['song_id'] = df['songs'].cat.codes



# Create item/user and user/item sparse matrices
sp_item_user = sp.csr_matrix((df['play count'].astype(float), (df['user_id'], df['song_id'])))
sp_user_item = sp.csr_matrix((df['play count'].astype(float), (df['song_id'], df['user_id'])))

# Calculate sparsity

matrix_size = sp_item_user.shape[0] * sp_item_user.shape[1]
non_zero_cells = len(sp_item_user.nonzero()[0])

sparsity = 1 - (non_zero_cells // matrix_size)


# Create ALS model

als_fitter = implicit.als.AlternatingLeastSquares(factors = 15, regularization = 0.1, iterations = 10)

# Assign alpha value (linear scaler) Collaborative Filtering for Implicit Feedback Datasets
# suggests 40 is good starting point

alpha_val = 40


# Scale item_user matrix with the alpha value

scaled_item_user = (sp_item_user * alpha_val).astype('double')




# Fit the scale

als_fitter.fit(scaled_item_user)


# Create user recommender - start with one user


recommendations = {}
for user in df['user_id']:
    recommendations[user_id] = als_fitter.recommend(user, sp_user_item)



    
    
    
    
 # Use grid search to optimize hyperparameters


# Create ALS model



# Grid search object

param_grid = {'num_factors': [10,20,40,60], 'regularization': [0.0, 0.0001, 0.001, 0.01, 0.1],
              'iterations': [20], 'alpha': [1,10,50,100,500]}

# Create implicit algorithm object

als_fitter = implicit.als.AlternatingLeastSquares()


# Create grid search object
gs = GridSearchCV(als_fitter, param_grid, measures = ['rmse', 'mae'], cv = 3,
                  verbose = True)

# Fit grid search object
gs.fit(new_train)


# print out results and best score based on root mean squared error
print(gs.cv_results_)

print(gs.best_score_['rmse'])


# Make new algorithm based off best parameters

als_fitter_2 = gs.best_estimator['rmse']

# Fit new model
als_fitter_2.fit(train_trip)





