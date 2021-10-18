import warnings
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import sys
import heapq


def create_dic_book_data(list_books_id, list_books_names):
    dic = {list_books_id[i]: list_books_names[i] for i in range(len(list_books_id))}
    return dic


def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    # replace anything lower than the cut off with 0
    arr[arr < smallest] = 0
    return arr


def get_key(val, dic):
    for key, value in dic.items():
        if val == value:
            return key


# Return top k movies that the user did not see!
def get_recommendations(predicted_ratings_row, data_matrix_row, k):
    dicNameBooks = create_dic_book_data(book_id, book_name)
    predicted_ratings_unrated = predicted_ratings_row
    predicted_ratings_unrated[~np.isnan((data_matrix_row))] = 0
    # highest_ratings = np.sort(predicted_ratings_unrated)[::1]
    # highest_k_pred = highest_ratings[0:k]
    idx = np.argsort(-predicted_ratings_unrated)
    sim_scores = idx[0:k]
    for index in sim_scores:
        old_index = get_key(index, dic_index_book)
        print("Original book id is: " + str(old_index) + " " + "The book name is: " + dicNameBooks[old_index])


# get book name and id
books = pd.read_csv('books.csv', encoding="ISO-8859-1")
book_name = books["title"]
book_id = books["book_id"]

# get users data
users = pd.read_csv('users.csv', encoding="ISO-8859-1")

# calculate the number of book and users in order to create matrix
ratings = pd.read_csv('ratings.csv')
n_users = ratings["user_id"].unique().shape[0]
n_books = ratings["book_id"].unique().shape[0]

# reindex books
i = 0
dic_index_book = {}
for old_ind in books["book_id"]:
    dic_index_book[old_ind] = i
    i += 1

# reindex users
j = 0
dic_index_user = {}
for old_ind in users["user_id"]:
    dic_index_user[old_ind] = j
    j += 1


data_matrix = np.empty((n_users, n_books))
data_matrix[:] = np.nan
for line in ratings.itertuples():
    # line[0] is index
    user = dic_index_user[line[1]]
    book = dic_index_book[line[2]]
    rating = line[3]
    data_matrix[user, book] = rating
df_matrix = pd.DataFrame(data_matrix)
mean_user_rating = np.nanmean(df_matrix, axis=1).reshape(-1, 1)
# Subtract the user rating average from each value
ratings_diff = (data_matrix - mean_user_rating)
# replace nan -> 0
ratings_diff[np.isnan(ratings_diff)] = 0


def build_CF_prediction_matrix(sim):
    if sim == "cosine":
        # calculate user x user similarity matrix
        user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
    elif sim == "euclidean":
        # calculate user x user similarity matrix
        user_similarity = 1 - pairwise_distances(ratings_diff, metric='euclidean')
    elif sim == "jaccard":
        # calculate user x user similarity matrix
        user_similarity = 1 - pairwise_distances(ratings_diff, metric='jaccard')
    else:
        print("your choice for imagination is not available")
    return user_similarity


def get_CF_recommendation(user_id, k):
    user_similarity = build_CF_prediction_matrix(sim)
    # For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
    # Note that the user has the highest similarity to themselves.
    user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])
    # user_id minus 1 because we save daya from user zero when the first user is 1
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    new_index_user = dic_index_user[user_id]
    predicted_ratings_row = pred[new_index_user]
    data_matrix_row = data_matrix[new_index_user]
    get_recommendations(predicted_ratings_row, data_matrix_row, k)


if __name__ == '__main__':
    sim = sys.argv[1]
    get_CF_recommendation(1, 10)
