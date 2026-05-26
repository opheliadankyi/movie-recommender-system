import os
import polars as pl
import numpy as np

import help_functions as H



def load_ratings_data(path, dataset_path, folder_name):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Ratings file not found in {path}")
    
    # Define experiment folder
    experiment_folder = os.path.abspath(os.path.join(path, folder_name, 'preprocess'))
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Load and preprocess ratings data
    # ratings = pl.read_csv(
    #     os.path.join(dataset_path, 'ratings.dat'),
    #     separator=':',
    #     has_header=False,
    #     encoding='latin1'
    # ).select([
    #     pl.col('column_1').alias('userId'),
    #     pl.col('column_3').alias('movieId'),
    #     pl.col('column_5').alias('rating'),
    #     pl.col('column_7').alias('timestamp')
    # ])
    ratings = pl.read_csv(
    os.path.join(dataset_path, 'ratings.csv')
    )
    
    # Group and extract mappings
    data_by_users = ratings.group_by('userId', maintain_order=True).agg(pl.col('movieId'), pl.col('rating'))
    data_by_movies = ratings.group_by('movieId', maintain_order=True).agg(pl.col('userId'), pl.col('rating'))
    
    map_users_to_ids = {userid: idx for idx, userid in enumerate(data_by_users['userId'].to_list())}
    map_movies_to_ids = {movieid: idx for idx, movieid in enumerate(data_by_movies['movieId'].to_list())}
    
    map_ids_to_users = {idx: userid for userid, idx in map_users_to_ids.items()}
    map_ids_to_movies = {idx: movieid for movieid, idx in map_movies_to_ids.items()}
    
    # Save mappings and data
    for name, data in zip(
        ['map_users_to_ids', 'map_movies_to_ids', 'map_ids_to_users', 'map_ids_to_movies', 'ratings_df'],
        [map_users_to_ids, map_movies_to_ids, map_ids_to_users, map_ids_to_movies, ratings]
    ):
        H.save_to_pickle(experiment_folder, data, name)
    
    # Clear memory
    del ratings, data_by_users, data_by_movies, map_users_to_ids, map_movies_to_ids, map_ids_to_users, map_ids_to_movies



def train_test_split(path, folder, seed=0, test_size=0.2):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ratings file not found at {path}")
    
    # Create experiment-specific directories
    os.makedirs(os.path.join(path, folder), exist_ok=True)
    
    # Load data and mappings
    ratings = H.load_from_pickle(os.path.join(path, folder, 'preprocess'), 'ratings_df')
    map_movie_to_id = H.load_from_pickle(os.path.join(path, folder, 'preprocess'), 'map_movies_to_ids')
    map_user_to_id = H.load_from_pickle(os.path.join(path, folder, 'preprocess'), 'map_users_to_ids')
    
    # Map user and movie IDs
    ratings = ratings.with_columns([
        pl.col("userId").replace_strict(map_user_to_id).alias("userIdx"),
        pl.col("movieId").replace_strict(map_movie_to_id).alias("movieIdx")
    ])
    
    # Random split for train-test using pl.when and pl.otherwise
    rng = np.random.default_rng(seed)
    ratings = ratings.with_columns(
        pl.when(pl.Series(rng.uniform(size=ratings.height)) > test_size)
          .then(pl.lit("train"))
          .otherwise(pl.lit("test"))
          .alias("split_label")
    )
    
    train_df = ratings.filter(pl.col("split_label") == "train").drop("split_label")
    test_df = ratings.filter(pl.col("split_label") == "test").drop("split_label")
    
    # Group data by users and movies
    users_train = train_df.group_by('userIdx', maintain_order=True).agg(pl.col('movieIdx'), pl.col('rating'))
    movies_train = train_df.group_by('movieIdx', maintain_order=True).agg(pl.col('userIdx'), pl.col('rating'))
    users_test = test_df.group_by('userIdx', maintain_order=True).agg(pl.col('movieIdx'), pl.col('rating'))
    movies_test = test_df.group_by('movieIdx', maintain_order=True).agg(pl.col('userIdx'), pl.col('rating'))
    
    # Convert to numpy arrays
    def extract_numpy(data):
        return data[:, 0].to_numpy(), data[:, 1:].to_numpy()
    
    users_indices_train, users_list_train = extract_numpy(users_train)
    movies_indices_train, movies_list_train = extract_numpy(movies_train)
    users_indices_test, users_list_test = extract_numpy(users_test)
    movies_indices_test, movies_list_test = extract_numpy(movies_test)
    
    # Save data to disk
    split_folder = os.path.join(path, folder, 'split')
    os.makedirs(split_folder, exist_ok=True)
    
    H.save_to_numpy(split_folder, users_list_train, 'users_list_train')
    H.save_to_numpy(split_folder, users_list_test, 'users_list_test')
    H.save_to_numpy(split_folder, movies_list_train, 'movies_list_train')
    H.save_to_numpy(split_folder, movies_list_test, 'movies_test')
    H.save_to_numpy(split_folder, users_indices_train, 'users_indices_train')
    H.save_to_numpy(split_folder, users_indices_test, 'users_indices_test')
    H.save_to_numpy(split_folder, movies_indices_train, 'movies_indices_train')
    H.save_to_numpy(split_folder, movies_indices_test, 'movies_indices_test')
    
    # Clean up memory
    del (ratings, test_df, train_df, users_list_train, users_list_test, 
         movies_list_train, movies_list_test, rng, map_user_to_id, map_movie_to_id)

    print("Train-test split completed and saved successfully.")

def load_movies_data(path, dataset_path, folder_name):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"movies file not found in {path}")
    
    # Define experiment folder
    experiment_folder = os.path.abspath(os.path.join(path, folder_name, 'preprocess'))
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Load movies data
    # movies = pl.read_csv(
    #     os.path.join(dataset_path, 'movies.dat'),
    #     separator=':',
    #     has_header=False,
    #     encoding='latin1',
    #     truncate_ragged_lines=True
    # ).select([
    #     pl.col('column_1').alias('movieId'),
    #     pl.col('column_3').alias('title'),
    #     pl.col('column_5').alias('genres')
    # ])
    movies = pl.read_csv(
    os.path.join(dataset_path, 'movies.csv')
    )
    # links_of_movie_ids = None    

    links_of_movie_ids = pl.read_csv(
    os.path.join(dataset_path, 'links.csv')
    )
    # Map movies to titles
    map_movies_to_titles = {movieid: title for movieid, title in zip(movies['movieId'].to_list(), movies['title'].to_list())}
    
    # Save mappings
    for name, data in zip(
        ['map_movies_to_titles', 'movies', 'links_of_movie_ids'],
        [map_movies_to_titles, movies, links_of_movie_ids]
    ):
        H.save_to_pickle(experiment_folder, data, name)
    
    # Delete variables to save memory
    del movies, map_movies_to_titles, links_of_movie_ids