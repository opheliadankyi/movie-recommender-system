from flask import Flask, render_template, request
import os
import requests
import urllib3
import pandas as pd
import help_functions as H

app = Flask(__name__)
# @app.route("/")
# def home():
#     return "Movie Recommender System is Live!"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "ca87348ba797784ff19f4e03918d2df4"

# Project paths
# path = r'G:\My Drive\AIMS\AML@SCALE\projects\RECOMMENDER-SYSTEM'
path = os.getcwd()
folder = 'files_25m'

# Load links.csv
# links_df = pd.read_csv(
#     os.path.join(path, 'data', 'ml-25m', 'links.csv')
# )

# # Load movie titles
# map_movies_to_titles = H.load_from_pickle(
#     os.path.join(path, folder, 'preprocess'),
#     'map_movies_to_titles'
# )

# # Map MovieLens movieId -> TMDB id
# movie_to_tmdb = dict(
#     zip(links_df['movieId'], links_df['tmdbId'])
# )

map_movies_to_titles = {}

movie_to_tmdb = {}

# Homepage categories
featured_categories = {

    "🔥 Action Movies": [
        "The Dark Knight",
        "Avengers",
        "John Wick",
        "Mad Max"
    ],

    "🧙 Fantasy Movies": [
        "Harry Potter",
        "Lord of the Rings",
        "Hobbit",
        "Fantastic Beasts"
    ],

    "💕 Romance Movies": [
        "Titanic",
        "The Notebook",
        "La La Land",
        "Pride and Prejudice"
    ],

    "🌌 Sci-Fi Movies": [
        "Interstellar",
        "Matrix",
        "Inception",
        "Star Wars"
    ]
}

# Get movie poster using TMDB ID
def get_movie_poster(tmdb_id):

    if pd.isna(tmdb_id):
        return "https://via.placeholder.com/300x450?text=No+Image"

    try:

        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={API_KEY}"

        response = requests.get(
            url,
            timeout=10,
            verify=False
        )

        data = response.json()

        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except:
        pass

    return "https://via.placeholder.com/300x450?text=No+Image"


@app.route('/', methods=['GET', 'POST'])
def home():

    movie_cards = []
    featured_movies = {}

    # Featured homepage sections
    for category, movies in featured_categories.items():

        featured_movies[category] = []

        for movie in movies:

            try:

                url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie}"

                response = requests.get(
                    url,
                    timeout=10,
                    verify=False
                )

                data = response.json()

                if data.get("results"):

                    poster_path = data["results"][0].get("poster_path")

                    if poster_path:

                        poster = f"https://image.tmdb.org/t/p/w500{poster_path}"

                        featured_movies[category].append({
                            'title': movie,
                            'poster': poster
                        })

            except:
                pass

    # Search recommendations
    if request.method == 'POST':

        movie_name = request.form['movie']

        # Find selected movie
        selected_movie_id = None

        for movie_id, title in map_movies_to_titles.items():

            if movie_name.lower() in title.lower():

                selected_movie_id = movie_id
                break

        # Generate recommendations
        if selected_movie_id is not None:

            # recommended_movies = H.recommend_movies(
            #     path,
            #     folder,
            #     selected_movie_id,
            #     5,
            #     10,
            #     0.01,
            #     0.01,
            #     0.01
            # )
            recommended_movies = [
                "Interstellar",
                "Inception",
                "The Matrix",
                "Avengers"
            ]
            print(recommended_movies)
            
            if recommended_movies is None:
                recommended_movies = []

            for title in recommended_movies:

                tmdb_id = None

                for movie_id, movie_title in map_movies_to_titles.items():

                    if movie_title == title:

                        tmdb_id = movie_to_tmdb.get(movie_id)
                        break

                poster = get_movie_poster(tmdb_id)

                movie_cards.append({
                    'title': title,
                    'poster': poster
                })

            movie_cards = movie_cards[:10]

    else:
        movie_name = ''

    return render_template(
        'index.html',
        movie_cards=movie_cards,
        featured_movies=featured_movies,
        movie_name=movie_name
    )


if __name__ == '__main__':
    app.run(debug=True)