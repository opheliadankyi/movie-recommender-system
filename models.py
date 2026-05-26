import os
import numpy as np
from numba import jit

############################## BIAS ONLY MODEL ###############################

@jit(nopython=True)
def user_bias_update(movie_biases, movie_indices, ratings, lamda, gamma):
    prediction = movie_biases[movie_indices]
    error = ratings - prediction
    return (lamda * np.sum(error)) / (lamda * len(movie_indices) + gamma)


@jit(nopython=True)
def movie_bias_update(user_biases, user_indices, ratings, lamda, gamma):
    prediction = user_biases[user_indices]
    error = ratings - prediction
    return (lamda * np.sum(error)) / (lamda * len(user_indices) + gamma)


@jit(nopython=True)
def calculate_squared_error(user_idx, user_biases, movie_biases, movie_indices, ratings):
    predictions = user_biases[user_idx] + movie_biases[movie_indices]
    return np.square(ratings - predictions)


@jit(nopython=True)
def calculate_total_squared_error(sq_error):
    return np.sum(sq_error)


def compute_biases_update(users_train, movies_train, users_indices_train, movies_indices_train,
                  user_biases, movie_biases, lamda, gamma):
    for idx, user_idx in enumerate(users_indices_train):
        movie_indices = users_train[idx][0]
        ratings = users_train[idx][1]
        user_biases[user_idx] = user_bias_update(movie_biases, movie_indices, ratings, lamda, gamma)
    
    for idx, movie_idx in enumerate(movies_indices_train):
        user_indices = movies_train[idx][0]
        ratings = movies_train[idx][1]
        movie_biases[movie_idx] = movie_bias_update(user_biases, user_indices, ratings, lamda, gamma)
    
    return user_biases, movie_biases


def calculate_loss(users_train, users_indices_train, user_biases, movie_biases, lamda, gamma):
    total_loss = 0.0
    for idx, user_idx in enumerate(users_indices_train):
        movie_indices = users_train[idx][0]
        ratings = users_train[idx][1]
        sq_error = calculate_squared_error(user_idx, user_biases, movie_biases, movie_indices, ratings)
        sum_sq_errors = calculate_total_squared_error(sq_error)
        total_loss += sum_sq_errors
    bias_penalty = np.sum(np.square(user_biases)) + np.sum(np.square(movie_biases))
    return 0.5 * lamda * total_loss + 0.5 * gamma * bias_penalty


def calculate_rmse(users_indices_train, users_train, users_indices_test, users_test, user_biases, movie_biases):
    squared_errors_train, squared_errors_test = [], []
    for idx, user_idx in enumerate(users_indices_train):
        movie_indices = users_train[idx][0]
        ratings = users_train[idx][1]
        sq_error = calculate_squared_error(user_idx, user_biases, movie_biases, movie_indices, ratings)
        squared_errors_train.extend(sq_error)
    
    for idx, user_idx in enumerate(users_indices_test):
        movie_indices = users_test[idx][0]
        ratings = users_test[idx][1]
        sq_error = calculate_squared_error(user_idx, user_biases, movie_biases, movie_indices, ratings)
        squared_errors_test.extend(sq_error)
    return np.sqrt(np.mean(squared_errors_train)), np.sqrt(np.mean(squared_errors_test))



######################## BIAS & factors ONLY MODEL ########################

@jit(nopython=True)
def user_bias_update_with_factors(user_idx, user_factors, movie_factors, 
                                     movie_biases, movie_indices, ratings, lamda, gamma):
    predictions = user_factors[user_idx] @ movie_factors[movie_indices].T + movie_biases[movie_indices]
    error = ratings - predictions
    return (lamda * np.sum(error)) / (lamda * len(movie_indices) + gamma)


@jit(nopython=True)
def user_factors_update(user_idx, movie_factors, user_biases, movie_biases, 
                          movie_indices, ratings, lamda, tau):
    predictions = user_biases[user_idx] + movie_biases[movie_indices]
    error = ratings - predictions
    other_term = lamda * movie_factors[movie_indices].T @ error
    inverse_term = lamda * movie_factors[movie_indices].T @ movie_factors[movie_indices] \
        + tau * np.eye(movie_factors.shape[1])
    return np.linalg.solve(inverse_term, other_term)


@jit(nopython=True)
def movie_bias_update_with_factors(movie_idx, user_factors, movie_factors, 
                                      user_biases, user_indices, ratings, lamda, gamma):
    predictions = user_factors[user_indices] @ movie_factors[movie_idx].T + user_biases[user_indices]
    error = ratings - predictions
    return (lamda * np.sum(error)) / (lamda * len(user_indices) + gamma)


@jit(nopython=True)
def movie_factors_update(movie_idx, user_factors, user_biases, movie_biases, 
                           user_indices, ratings, lamda, tau):
    predictions = movie_biases[movie_idx] + user_biases[user_indices]
    error = ratings - predictions
    other_term = lamda * user_factors[user_indices].T @ error
    inverse_term = lamda * user_factors[user_indices].T @ user_factors[user_indices] \
        + tau * np.eye(user_factors.shape[1])
    return np.linalg.solve(inverse_term, other_term)


@jit(nopython=True)
def calculate_squared_error_with_factors(user_idx, user_biases, user_factors, 
                                            movie_biases, movie_factors, movie_indices, ratings):
    predictions = user_factors[user_idx] @ movie_factors[movie_indices].T \
                  + user_biases[user_idx] + movie_biases[movie_indices]
    return np.square(ratings - predictions)


@jit(nopython=True)
def calculate_total_squared_error(sq_error):
    return np.sum(sq_error)


def compute_user_biases_and_factors(users_train, users_indices_train, user_biases, 
                                       user_factors, movie_factors, movie_biases, 
                                       lamda, gamma, tau):
    for idx, user_idx in enumerate(users_indices_train):
        movie_indices = users_train[idx][0]
        ratings = users_train[idx][1]
        user_biases[user_idx] = user_bias_update_with_factors(
            user_idx, user_factors, movie_factors, movie_biases, movie_indices, ratings, lamda, gamma
        )
        user_factors[user_idx] = user_factors_update(
            user_idx, movie_factors, user_biases, movie_biases, movie_indices, ratings, lamda, tau
        )
    return user_biases, user_factors


def compute_movie_biases_and_factors(movies_train, movies_indices_train, user_biases, 
                                        user_factors, movie_factors, movie_biases, 
                                        lamda, gamma, tau):
    for idx, movie_idx in enumerate(movies_indices_train):
        user_indices = movies_train[idx][0]
        ratings = movies_train[idx][1]
        movie_biases[movie_idx] = movie_bias_update_with_factors(
            movie_idx, user_factors, movie_factors, user_biases, user_indices, ratings, lamda, gamma
        )
        movie_factors[movie_idx] = movie_factors_update(
            movie_idx, user_factors, user_biases, movie_biases, user_indices, ratings, lamda, tau
        )
    return movie_biases, movie_factors


def calculate_loss_with_factors(users_train, users_indices_train, user_biases, user_factors, 
                                   movie_biases, movie_factors, lamda, gamma, tau):
    total_loss = 0.0
    for idx, user_idx in enumerate(users_indices_train):
        movie_indices = users_train[idx][0]
        ratings = users_train[idx][1]
        sq_error = calculate_squared_error_with_factors(
            user_idx, user_biases, user_factors, movie_biases, movie_factors, movie_indices, ratings
        )
        sum_sq_errors = calculate_total_squared_error(sq_error)
        total_loss += sum_sq_errors
    bias_penalty = 0.5 * gamma * (np.sum(np.square(user_biases)) + np.sum(np.square(movie_biases)))
    factor_penalty = 0.5 * tau * (np.sum(np.square(user_factors)) + np.sum(np.square(movie_factors)))
    return 0.5 * lamda * total_loss + bias_penalty + factor_penalty


def calculate_rmse_with_factors(users_indices_train, users_train, users_indices_test, users_test, user_biases, user_factors, 
                                   movie_biases, movie_factors):
    squared_errors_train, squared_errors_test = [], []
    for idx, user_idx in enumerate(users_indices_train):
        movie_indices = users_train[idx][0]
        ratings = users_train[idx][1]
        sq_error = calculate_squared_error_with_factors(
            user_idx, user_biases, user_factors, movie_biases, movie_factors, movie_indices, ratings
        )
        squared_errors_train.extend(sq_error)
        
    for idx, user_idx in enumerate(users_indices_test):
        movie_indices = users_test[idx][0]
        ratings = users_test[idx][1]
        sq_error = calculate_squared_error_with_factors(
            user_idx, user_biases, user_factors, movie_biases, movie_factors, movie_indices, ratings
        )
        squared_errors_test.extend(sq_error)
        
    return np.sqrt(np.mean(squared_errors_train)), np.sqrt(np.mean(squared_errors_test))
