"""
Advanced Recommender Systems
"""
from math import sqrt
from numpy import abs as npabs
from numpy import zeros, newaxis, array, diag, dot
from pandas import read_csv, DataFrame, merge
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

# Getting the Data
column_names = ["user_id", "item_id", "rating", "timestamp"]
df = DataFrame(read_csv("u.data", sep="\t", names=column_names))
print(df.head())
movie_titles = DataFrame(read_csv("Movie_Id_Titles"))
print(movie_titles.head())
df = merge(df, movie_titles, on="item_id")
print(df.head())
n_users = df.user_id.nunique()
n_items = df.item_id.nunique()
print("Num. of Users: " + str(n_users))
print("Num of Movies: " + str(n_items))

# Train Test Split
train_data, test_data = train_test_split(df, test_size=0.25)

# Create two user-item matrices, one for training and another for testing
train_data_matrix = zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
test_data_matrix = zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]
user_similarity = pairwise_distances(train_data_matrix, metric="cosine")
item_similarity = pairwise_distances(train_data_matrix.T, metric="cosine")


def predict(ratings, similarity, types="user"):
    """
    Use newaxis so that mean_user_rating has same format as ratings
    """
    if types == "user":
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = ratings - mean_user_rating[:, newaxis]
        pred = (
            mean_user_rating[:, newaxis]
            + similarity.dot(ratings_diff) / array([npabs(similarity).sum(axis=1)]).T
        )
    elif types == "item":
        pred = ratings.dot(similarity) / array([npabs(similarity).sum(axis=1)])
    return pred


item_prediction = predict(train_data_matrix, item_similarity, types="item")
user_prediction = predict(train_data_matrix, user_similarity, types="user")


def rmse(prediction, ground_truth):
    """
    Root Mean Squared Error
    """
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


print("User-based CF RMSE: " + str(rmse(user_prediction, test_data_matrix)))
print("Item-based CF RMSE: " + str(rmse(item_prediction, test_data_matrix)))

# Model-based Collaborative Filtering
sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
print("The sparsity level of MovieLens100K is " + str(sparsity * 100) + "%")

# Get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k=20)
s_diag_matrix = diag(s)
X_pred = dot(dot(u, s_diag_matrix), vt)
print("User-based CF MSE: " + str(rmse(X_pred, test_data_matrix)))
