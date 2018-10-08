#! python 3
# msd_imp.py - recommendation system for million song challenge using implicit

## Import necessary libraries

import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as ms
import implicit
import random
from sklearn import metrics
from sklearn.model_selection import GridSearchCV






# Training data
train = pd.read_csv('train_triplets.txt', delimiter = '\t', header = None)
train.columns = ['users', 'songs', 'play count']

# Test data
test = pd.read_csv('kaggle_visible_evaluation_triplets.txt', delimiter = '\t', header = None, names = ['users', 'songs', 'play count'])

# Combine the two dataframes, as this an implicit collaborative filter so we need all user/item interactions for matrix factorization

df = pd.concat([train, test], ignore_index = True)

# Make the userid and songid strings
df['songs'] = df['songs'].astype("category")
df['users'] = df['users'].astype("category")


# Inspect data

df.head()
df.shape

# How many unique users

print(len(df['users'].drop_duplicates()))

# How many unique songs

print(len(df['songs'].drop_duplicates()))

# Check for null values

print(df['play count'].isnull().sum())

# Check the max and min of each column

play_count_max = df['play count'].max()
play_count_min = df['play count'].min()

# pivot data to determine most popular songs

song_pivot = df.pivot_table(index = 'songs',
                      aggfunc = sum)


# Check the most popular songids

song_popularity = df.songs.value_counts()


# Plot histogram for distribution
sns.set()
_ = plt.hist(df['play count'], bins = 10, range = [play_count_min, 40], normed = True)
_ = plt.xlabel('Play counts')
_ = plt.ylabel('Counts')
plt.show()

# Plot boxplot

_ = sns.boxplot(df['play count'])
plt.show()

df['user_id'] = df['users'].cat.codes
df['song_id'] = df['songs'].cat.codes


# Calculate density

sparsity = 1-(df.shape[0] / (df.user_id.unique().shape[0] * df.song_id.unique().shape[0]))

# Our sparsity is over 99.5%, which is the minimum threshold for implicit ALS to be effective
# we will remove songs under x listens


def activity_thresh(df, user_min, song_min):
    while True:
        start_dim = df.shape[0]
        song_counts = df.groupby('user_id').song_id.count()
        df = df[~df.user_id.isin(song_counts[song_counts < song_min].index.tolist())]
        user_counts = df.groupby('song_id').user_id.count()
        df = df[~df.song_id.isin(user_counts[user_counts < user_min].index.tolist())]
        end_dim = df.shape[0]
        if start_dim == end_dim:
            break
    
    n_users = df.user_id.unique().shape[0]
    n_items = df.song_id.unique().shape[0]
    sparsity = 1- (float(df.shape[0]) / float(n_users*n_items))
    print('Number of users: {}'.format(n_users))
    print('Number of songs: {}'.format(n_items))
    print('Sparsity: {:.5%}'.format(sparsity))
    return df, sparsity


for i in range(5,50):
    print('Min user and song count: {}'.format(i))
    _, sparsity = activity_thresh(df, i, i)
    if sparsity < 0.995:
        break
    else:
        continue


df_red, _ = activity_thresh(df, 25, 25)



# Make user and song into coded factors and convert to sparse matrix for processing





# Create item/user and user/item sparse matrices
sp_item_user = sp.csr_matrix((df_red['play count'].astype(float), (df_red['user_id'], df_red['song_id'])))
sp_user_item = sp.csr_matrix((df_red['play count'].astype(float), (df_red['song_id'], df_red['user_id'])))







# Masking a certain percentage of song/user interactions and then check how many songs were eventually listned to which were recommended


def train_test_mask(sp_matrix, test_split = 0.2):
  
    test_set = sp_matrix.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = sp_matrix.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(42) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(test_split*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # 

song_train, song_test, user_lst = train_test_mask(sp_item_user)




# Use alternating Least Squares to create recommendation algorithm
alpha = 15
rec = implicit.als.AlternatingLeastSquares(factors = 20, regularization = 0.1,
                                           iterations = 20)
rec.fit(song_train * alpha)


user_factors, item_factors = rec.item_factors, rec.user_factors




def auc_score(pred, test):

    fpr, tpr, thresholds = metrics.roc_curve(test, pred)
    return metrics.auc(fpr, tpr)  






def area_under_curve(training_set, altered_users, predictions, test_set):

    
    
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had an item altered
        print("Loop {}".format(altered_users[user]))
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  
   # Return the mean AUC rounded to three decimal places for both test a


area_under_curve(song_train, user_lst, 
              [sp.csr_matrix(user_factors), sp.csr_matrix(item_factors.transpose())], song_test)







