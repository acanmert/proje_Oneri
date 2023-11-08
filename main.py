import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_data =pd.read_csv('data.csv')

movie_data['combined_features'] = movie_data.apply(lambda row: ' '.join([str(row['title']), str(row['genre']), str(row['director']), str(row['actors'])]), axis=1)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movie_data['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_movie_recommendations(movie_title, top_n=10):
    # Find the index of the movie in the dataset
    movie_index = movie_data[movie_data['title'] == movie_title].index[0]

    # Get the similarity scores for the movie
    similarity_scores = list(enumerate(cosine_sim[movie_index]))

    # Sort movies based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top recommendations
    top_recommendations = similarity_scores[1:top_n + 1]

    # Retrieve movie titles for the recommended movie indices
    recommended_movies = [movie_data['title'][movie[0]] for movie in top_recommendations]

    return recommended_movies, [score[1] for score in top_recommendations]

movie_title = input("Enter a movie title: ")

recommendations, similarity_scores = get_movie_recommendations(movie_title, top_n=10)

plt.figure(figsize=(10, 6))
plt.barh(recommendations, similarity_scores)
plt.xlabel('Cosine Similarity')
plt.ylabel('Movie Title')
plt.title(f"Top 10 Recommended Movies for '{movie_title}' based on Cosine Similarity")
plt.tight_layout()
plt.show()

output_file = 'output.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(f"Top 10 Recommended Movies for '{movie_title}':\n")
    for movie in recommendations:
        file.write(movie + '\n')


print(f"Recommended movies for '{movie_title}' have been written to '{output_file}'.")
