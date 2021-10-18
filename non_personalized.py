import pandas as pd
import sys

# display full Dataframe without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


read_metadata_rating = pd.read_csv("ratings.csv", encoding="ISO-8859-1")
read_metadata_users = pd.read_csv("users.csv", encoding="ISO-8859-1")
read_metadata_books = pd.read_csv("books.csv", encoding="ISO-8859-1")

data = {"book_id": read_metadata_books["book_id"]}
# create df
df = pd.DataFrame(data)
counts_per_book = read_metadata_rating.pivot_table(index=['book_id'], aggfunc='size')
df.insert(1, "counts", counts_per_book.values)
sum_rating_per_book = read_metadata_rating.groupby("book_id")["rating"].sum()
df.insert(2, "sum_rating", sum_rating_per_book.values)
df.insert(3, "average_rating", sum_rating_per_book.values/counts_per_book.values)
df.insert(4, "book_name", read_metadata_books["title"])


# This function return minimum numbers of votes that needed in order to consider them in Non personalized recommenders
def min_votes(df, title):
    min_v = df[title].quantile(0.9)
    return min_v


# keys are user id and values are the data that related to the user
def create_dic_users_data(list_users_id, list_data_on_users, size):
    dic = {list_users_id[i]: list_data_on_users[i] for i in range(size)}
    return dic


def cut_data_by_min_v(minimum_votes, metadata, title):
    q_metadata = metadata.copy().loc[metadata[title] >= minimum_votes]
    return q_metadata


def cut_data_by_string(string_data, df, title):
    cut_df = df.copy().loc[df[title] == string_data]
    return cut_df


def cut_data_by_range_num(x, y, metadata, title):
    q_metadata = metadata.copy().loc[(metadata[title] >= x) & (metadata[title] <= y)]
    return q_metadata


def weighted_average_rating(v, r, m, c):
    w_r = ((v / (v + m)) * r) + ((m / (v + m)) * c)
    return w_r


def get_simply_recommendation(k):
    k = int(k)
    average_rating_score = df["average_rating"].mean()
    m_vote = min_votes(df, "counts")
    cut_df = cut_data_by_min_v(m_vote, df, "counts")
    # add column of weighted_rating for each line
    cut_df["weighted_rating"] = cut_df.apply(lambda row: weighted_average_rating(row["counts"], row["average_rating"], m_vote, average_rating_score), axis=1)
    # sort by weighted rating
    cut_df = cut_df.sort_values("weighted_rating", ascending=False)
    print(cut_df[['book_id', 'book_name', 'counts', 'average_rating', 'weighted_rating']].head(k))


# dic that connect between book_id to book_name
dic_name_book = create_dic_users_data(read_metadata_books["book_id"],read_metadata_books["title"], len(read_metadata_books["book_id"]))


def get_simply_place_recommendation(place, k):
    k = int(k)
    data_location = {"user_id": read_metadata_rating["user_id"], "book_id": read_metadata_rating["book_id"], "rating": read_metadata_rating["rating"]}
    df_location = pd.DataFrame(data_location)
    dic_location_users = create_dic_users_data(read_metadata_users["user_id"], read_metadata_users["location"], max(read_metadata_rating["user_id"]))
    # for each book, we know the user that rate - and now we add the location of this user
    df_location["location"] = df_location.apply(lambda row: dic_location_users[row["user_id"]], axis=1)
    # cut all users that not match to the relevant place\location is given
    cut_df_location = cut_data_by_string(place, df_location, "location")
    # count all users from location that voted per book
    counts_per_book_loc = cut_df_location.pivot_table(index=['book_id'], aggfunc='size')
    data_loc_summery = {"book_id": counts_per_book_loc.index, "counts": counts_per_book_loc.values}
    df_loc_summery = pd.DataFrame(data_loc_summery)
    sum_rating_per_book_loc = cut_df_location.groupby("book_id")["rating"].sum()
    df_loc_summery.insert(2, "sum_rating", sum_rating_per_book_loc.values)
    df_loc_summery.insert(3, "average_rating", sum_rating_per_book_loc.values / counts_per_book_loc.values)
    df_loc_summery.insert(4, "book_name", lambda row: dic_name_book[row["book_id"]])
    df_loc_summery["book_name"] = df_loc_summery.apply(lambda row: dic_name_book[row["book_id"]], axis=1)
    average_rating_score = df_loc_summery["average_rating"].mean()
    m_vote = min_votes(df_loc_summery, "counts")
    df_loc_summery = cut_data_by_min_v(m_vote, df_loc_summery, "counts")
    df_loc_summery["score"] = df_loc_summery.apply(lambda row: weighted_average_rating(row["counts"], row["average_rating"], m_vote, average_rating_score), axis=1)
    # sort by weighted rating
    df_loc_summery = df_loc_summery.sort_values("score", ascending=False)
    print(df_loc_summery[['book_id', 'book_name', 'counts', 'average_rating', 'score']].head(k))


def get_simply_age_recommendation(age, k):
    age = int(age)
    k = int(k)
    range_x1 = age - age % 10 + 1
    range_y0 = range_x1 + 9
    data_location = {"user_id": read_metadata_rating["user_id"], "book_id": read_metadata_rating["book_id"], "rating": read_metadata_rating["rating"]}
    df_age = pd.DataFrame(data_location)
    dic_age_users = create_dic_users_data(read_metadata_users["user_id"], read_metadata_users["age"], max(read_metadata_rating["user_id"]))
    # for each book, we know the user that rate - and now we add the location of this user
    df_age["age"] = df_age.apply(lambda row: dic_age_users[row["user_id"]], axis=1)
    # cut all users that not match to the relevant age is given
    cut_df_location = cut_data_by_range_num(range_x1, range_y0, df_age, "age")
    # count all users from location that voted per book
    counts_per_book_age = cut_df_location.pivot_table(index=['book_id'], aggfunc='size')
    data_age_summery = {"book_id": counts_per_book_age.index, "counts": counts_per_book_age.values}
    df_age_summery = pd.DataFrame(data_age_summery)
    sum_rating_per_book_loc = cut_df_location.groupby("book_id")["rating"].sum()
    df_age_summery.insert(2, "sum_rating", sum_rating_per_book_loc.values)
    df_age_summery.insert(3, "average_rating", sum_rating_per_book_loc.values / counts_per_book_age.values)
    df_age_summery["book_name"] = df_age_summery.apply(lambda row: dic_name_book[row["book_id"]], axis=1)
    average_rating_score = df_age_summery["average_rating"].mean()
    m_vote = min_votes(df_age_summery, "counts")
    df_age_summery = cut_data_by_min_v(m_vote, df_age_summery, "counts")
    df_age_summery["score"] = df_age_summery.apply(lambda row: weighted_average_rating(row["counts"], row["average_rating"], m_vote, average_rating_score), axis=1)
    # sort by weighted rating
    df_age_summery = df_age_summery.sort_values("score", ascending=False)
    print(df_age_summery[['book_id', 'book_name', 'counts', 'average_rating', 'score']].head(k))


if __name__ == '__main__':
    place_or_age = sys.argv[1]
    k_Recommendations = sys.argv[2]
    #get_simply_recommendation(k_Recommendations)
    #get_simply_place_recommendation(place_or_age, k_Recommendations)
    #get_simply_age_recommendation(28, 10)
    get_simply_age_recommendation(place_or_age, k_Recommendations)
