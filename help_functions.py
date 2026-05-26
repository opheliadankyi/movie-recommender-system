import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import pickle

fontsize = 18
params = {
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize
}
plt.rcParams.update(params)

def save_to_pickle(path, data, filename):
    with open(os.path.join(path, f'{filename}.pkl'), 'wb') as f:
        pickle.dump(data, f)

    
def load_from_pickle(path, filename, log=True):
    with open(os.path.join(path, f'{filename}.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data

def save_to_numpy(path, data, filename, log=True):
    np.save(os.path.join(path, f'{filename}.npy'), data, allow_pickle=True)
    
def load_from_numpy(path, filename, log=True):
    return np.load(os.path.join(path, f'{filename}.npy'), allow_pickle=True)


def plot_loss_and_rmse(path, folder, loss_history, rmse_history, model_type=''):
    # Create necessary directories
    os.makedirs(os.path.join(path, folder, 'plots'), exist_ok=True)
    image_folder = os.path.join(path, folder, 'plots')
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Train Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    #plt.title('Loss over Iterations')
    plt.legend()
    loss_filename = f"loss_plot_{model_type}.pdf"
    plt.savefig(os.path.join(image_folder, loss_filename), format='pdf', dpi=3000)
    plt.show()
    
    # Plot RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_history['train'], label='Train RMSE', color='red')
    plt.plot(rmse_history['test'], label='Test RMSE', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    #plt.title('RMSE over Iterations')
    plt.legend()
    rmse_filename = f"rmse_plot_{model_type}.pdf"
    plt.savefig(os.path.join(image_folder, rmse_filename), format='pdf', dpi=3000)
    plt.show()
    print(f"Plots saved in {image_folder}")

def plot_rating_distribution(path, folder):
    # Create necessary directories
    os.makedirs(os.path.join(path, folder, 'plots'), exist_ok=True)
    image_folder = os.path.join(path, folder, 'plots')
    
    ratings_df = load_from_pickle(os.path.join(path, folder, 'preprocess'), 'ratings_df')
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_df['rating'], bins=10, kde=False)
    #plt.title('Rating Distribution of the MovieLens Dataset')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    filename = f"ratings_distribution"
    plt.savefig(os.path.join(image_folder, filename), format ='pdf', dpi=3000)
    plt.show()
    del ratings_df
    
def plot_powerlaw_distribution(path, folder):
    # Create necessary directories
    os.makedirs(os.path.join(path, folder, 'plots'), exist_ok=True)
    image_folder = os.path.join(path, folder, 'plots')
    

    # Calculate the number of ratings per user and per movie
    ratings_df = load_from_pickle(os.path.join(path, folder, 'preprocess'), 'ratings_df')
    ratings_per_user = ratings_df.group_by('userId').agg(pl.count().alias('count')).sort('count')
    ratings_per_movie = ratings_df.group_by('movieId').agg(pl.count().alias('count')).sort('count')
    
    # Calculate the frequency of each count value (equivalent to value_counts in Pandas)
    user_count_distribution = (ratings_per_user.group_by('count').agg(pl.count().alias('frequency')).sort('count'))
    movie_count_distribution = (ratings_per_movie.group_by('count').agg(pl.count().alias('frequency')).sort('count'))

    # Plot the distributions on a log-log scale
    plt.figure(figsize=(10, 6))
    plt.loglog(user_count_distribution['count'], user_count_distribution['frequency'], marker='o', linestyle='None', alpha=0.6, label='Users')
    plt.loglog(movie_count_distribution['count'], movie_count_distribution['frequency'], marker='o', linestyle='None', alpha=0.6, label='Movies')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Frequency')
    #plt.title('Log-Log Scale (Power Law disttibution of the 25 million Movielens dataset)')
    plt.legend()
    filename = f"powerlaw_distribution"
    plt.savefig(os.path.join(image_folder, filename), format ='pdf', dpi=3000)
    plt.show()
    
    del ratings_df, ratings_per_movie, ratings_per_user, user_count_distribution, movie_count_distribution
    
def recommend_movies(path, folder, movie_id, movie_rating, top_n, lamda, gamma, tau):
    # Define paths for loading saved files
    process_path = os.path.join(path, folder, 'preprocess')
    model_files = os.path.join(path, folder, 'model')
    
    # Load model files
    movies_factors = load_from_numpy(model_files, 'movies_factors')
    map_movies_to_titles = load_from_pickle(process_path, 'map_movies_to_titles')
    map_movies_to_ids = load_from_pickle(process_path,'map_movies_to_ids')
    map_ids_to_movies = load_from_pickle(process_path,'map_ids_to_movies')
    movies_biases = load_from_numpy(model_files, 'movies_biases')
    
    # Update user embeddings
    movie_idx = map_movies_to_ids[movie_id]
    user_movie_factors = movies_factors[movie_idx]
    user_movie_bias = movies_biases[movie_idx]
    user_factors = np.zeros((1, movies_factors.shape[1]))
    user_bias = lamda * (movie_rating - (user_factors @ user_movie_factors + user_movie_bias)) / (lamda  + gamma)
    
    for _ in range(5):
        inverse_term = lamda * (user_movie_factors.T @ user_movie_factors) + tau * np.eye(movies_factors.shape[1])
        other_term = user_movie_factors.T * (movie_rating - (user_movie_bias + user_bias))
        user_factors = np.linalg.solve(inverse_term, other_term)
        user_bias = lamda * (movie_rating - (user_factors.T @ user_movie_factors + user_movie_bias)) / (lamda  + lamda)
    
    # Compute scores for all items
    scores = (
        np.inner(user_factors, movies_factors) +
        0.05 * movies_biases
    )

    # Get  the indices of the top_n scores
    top_indices = np.argsort(scores)[-top_n:][::-1]

    # Map back to movie IDs
    top_movie_ids = [map_ids_to_movies[i] for i in top_indices]

    # Map back to movie Titles
    item_titles = [map_movies_to_titles[item_id] for item_id in top_movie_ids]
    
    movie_title = map_movies_to_titles[movie_id]

    print(f"Since you liked '{movie_title}', \nYou may also like: \n")
    for i, title in enumerate(item_titles):
        print(f"{i+1}. {title}")

    return item_titles