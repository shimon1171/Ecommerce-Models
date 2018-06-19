from sklearn  import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from math import sqrt


# Function to predict ratings
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


def test1():
    filename = 'dataset.csv'
    df = pd.read_csv(filename)
    # small_data = df.values.tolist()
    train_data, test_data = cv.train_test_split(df, test_size=0.2)

    # Create two user-item matrices, one for training and another for testing
    train_data_matrix = train_data.as_matrix(columns=['user', 'movie', 'rate'])
    test_data_matrix = test_data.as_matrix(columns=['user', 'movie', 'rate'])

    # User Similarity Matrix
    user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
    user_correlation[np.isnan(user_correlation)] = 0
    # Item Similarity Matrix
    item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
    item_correlation[np.isnan(item_correlation)] = 0

    # Predict ratings on the training data with both similarity score
    user_prediction = predict(train_data_matrix, user_correlation, type='user')
    item_prediction = predict(train_data_matrix, item_correlation, type='item')
    # RMSE on the train data
    print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
    print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))






if __name__ == '__main__':
    test1()