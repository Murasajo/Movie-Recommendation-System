import streamlit as st
import pickle
from surprise import SVD, Dataset, Reader
import requests

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=e636f249845ba0a95710fdd5ad4852e5"
    response = requests.get(url)
    data = response.json()

    if 'poster_path' in data and data['poster_path'] is not None:
        poster_path = data['poster_path']
        full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
    else:
        full_path = "https://via.placeholder.com/500x750?text=No+Poster+Available"

    return full_path

movies = pickle.load(open("movies_list.pkl", 'rb'))
svd = pickle.load(open("svd_model.pkl", 'rb'))
movies_list = movies['title'].values

st.header("Movie Recommender System")


import streamlit.components.v1 as components

imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

imageUrls = [
    fetch_poster(663),
    fetch_poster(299536),
    fetch_poster(2391),
    fetch_poster(2282),
    fetch_poster(429422),
    fetch_poster(9722),
    fetch_poster(13972),
    fetch_poster(240),
    fetch_poster(155),
    fetch_poster(598),
    fetch_poster(914),
    fetch_poster(255709),
    fetch_poster(572154)
]

imageCarouselComponent(imageUrls=imageUrls, height=200)


selectvalue = st.selectbox("Select movie from dropdown", movies_list, key="movie_selector")


def get_movie_id(movie_name):
    return movies[movies['title'] == movie_name]['movieId'].values[0]

def get_movie_name(movie_id):
    return movies[movies['movieId'] == movie_id]['title'].values[0]

def get_recommendations_by_movie(movie_name, N=5):
    # Get the movie ID for the given movie name
    movie_id = get_movie_id(movie_name)
    
    # Get all users who have rated the given movie
    user_ids = movies[movies['movieId'] == movie_id]['userId'].unique()
    
    # Get a list of all movie ids
    all_movie_ids = movies['movieId'].unique()
    
    # Create a dictionary to store the estimated ratings for each movie
    movie_ratings = {movie_id: 0 for movie_id in all_movie_ids}
    
    # Get the count of ratings for normalization
    rating_count = {movie_id: 0 for movie_id in all_movie_ids}
    
    # Predict the ratings for all movies for each user
    for user_id in user_ids:
        for movie_id in all_movie_ids:
            prediction = svd.predict(user_id, movie_id)
            movie_ratings[movie_id] += prediction.est
            rating_count[movie_id] += 1
    
    # Normalize the ratings by dividing by the count of ratings
    for movie_id in movie_ratings:
        if rating_count[movie_id] > 0:
            movie_ratings[movie_id] /= rating_count[movie_id]
    
    # Sort the movies by the estimated rating in descending order
    sorted_movie_ratings = sorted(movie_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top N recommendations (excluding the input movie itself)
    top_recommendations = [get_movie_name(movie_id) for movie_id, rating in sorted_movie_ratings if movie_id != get_movie_id(movie_name)][:N]
    recommendation_posters = [fetch_poster(movie_id) for movie_id in [get_movie_id(name) for name in top_recommendations]]
    
    return top_recommendations, recommendation_posters


if st.button("Show recommend"):
    movie_names, movie_posters = get_recommendations_by_movie(selectvalue)

    columns = st.columns(5)
    for idx, col in enumerate(columns):
        with col:
            st.text(movie_names[idx])
            st.image(movie_posters[idx])