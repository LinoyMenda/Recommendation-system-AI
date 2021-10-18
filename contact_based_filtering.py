import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

metadata = pd.read_csv('books.csv', low_memory=False, encoding="ISO-8859-1")
features_metadata = {"book_id": metadata["book_id"], "authors": metadata["authors"],
                     "original_publication_year": metadata["original_publication_year"],
                     "original_title": metadata["original_title"], "language_code": metadata["language_code"]}
df_features_metadata = pd.DataFrame(features_metadata)


def create_soup(x):
    return ' ' + ''.join(str(x['authors'])) + ' ' + ''.join(str(x['original_publication_year'])) + ' ' + ''.join(str(x['original_title'])) + ' ' + ''.join(str(x['language_code']))


# Create a new soup feature
df_features_metadata['soup'] = df_features_metadata.apply(create_soup, axis=1)


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df_features_metadata['soup'])
#print(count_matrix.shape)

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# Reset index of your main DataFrame and construct reverse mapping as before
df_features_metadata = df_features_metadata.reset_index()
indices = pd.Series(df_features_metadata.index, index=df_features_metadata['original_title'])
#print(indices[:10])


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[title]
    if len(idx) > 1:
        idx = idx[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (the first is the movie we asked)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df_features_metadata['original_title'].iloc[book_indices]


if __name__ == '__main__':
    print(get_recommendations("Twilight"))
