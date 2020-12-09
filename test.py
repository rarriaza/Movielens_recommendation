import pickle

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras import Sequential
from pathlib import Path
import pandas as pd
import datetime
import h5py
import argparse

from data_preprocessing import preprocessing
from train import create_model


def load_df_movie(file_path): #"ml-100k/u.item"
    header = ["movie id", "movie title", "release date", "video release date",
              "IMDb URL", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    item_df = pd.read_csv(file_path, sep="|", encoding='latin-1', names=header)
    item_df = item_df.drop(["release date", "video release date",
              "IMDb URL", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"], axis=1)
    item_df = item_df.set_index("movie id")
    return item_df


def create_movies(person_id, user_df, movies_df, reviewed_movies):
    n_mov_cols = len(movies_df.columns)
    n_users_cols = len(user_df.columns)
    n_features = n_mov_cols + n_users_cols
    movies_seen = set([int(j) for j in reviewed_movies[person_id]])
    n_movies = len(movies_df) - len(movies_seen)

    x = np.zeros((n_movies, n_features))
    ids = []
    i = 0
    movies_iter = set(movies_df.index).difference(movies_seen)
    for movie_id in movies_iter:
        ids += [movie_id]
        x[i, 0: n_users_cols] = user_df.loc[person_id]
        x[i, n_users_cols:] = movies_df.loc[movie_id]
        i += 1
    return x, ids


def load_info():
    xtrain = np.load("Xtrain.npy")
    ytrain = np.load("Ytrain.npy")
    xtest = np.load("Xtest.npy")
    ytest = np.load("Ytest.npy")
    movies_df = pd.read_csv("movies_df.csv", index_col=0)
    user_df = pd.read_csv("user_df.csv", index_col=0)
    reviewed_movies = np.load("reviewed_movies.npy", allow_pickle=True).tolist()
    id2title = load_df_movie("ml-100k/u.item")
    return xtrain, ytrain, xtest, ytest, movies_df, user_df, reviewed_movies, id2title


def get_ranking_DL(person_id):
    path_checkpoint = "checkpoints/model_weights-24-0.04.h5"
    xtrain, ytrain, xtest, ytest, movies_df, user_df, reviewed_movies, id2title = load_info()
    model = create_model(xtrain.shape[1])
    random_pred = model.predict(xtest[0:2, :])
    model.load_weights(path_checkpoint)
    x, ids = create_movies(person_id, user_df, movies_df, reviewed_movies)
    pred = model.predict(x)
    ranking = np.argsort(pred.ravel())[::-1]
    print("Ranking of 5 recommended movies for user {} with Neural Network:".format(person_id))
    for i in range(5):
        movie_id = ids[ranking[i]]
        movie_title = id2title.loc[movie_id]["movie title"]
        print("Option {} - Movie: {} - {}, Possible score: {}".format(i, movie_id, movie_title, pred[ranking[i]] * 5))


def get_ranking_RF(person_id):
    path_checkpoint = "checkpoints/rf_model.pkl"
    xtrain, ytrain, xtest, ytest, movies_df, user_df, reviewed_movies, id2title = load_info()
    model = pickle.load(open(path_checkpoint, "rb"))
    x, ids = create_movies(person_id, user_df, movies_df, reviewed_movies)
    pred = model.predict(x)
    ranking = np.argsort(pred.ravel())[::-1]
    print("\n")
    print("Ranking of 5 recommended movies for user {} with Random Forest:".format(person_id))
    for i in range(5):
        movie_id = ids[ranking[i]]
        movie_title = id2title.loc[movie_id]["movie title"]
        print("Option {} - Movie: {} - {}, Possible score: {}".format(i, movie_id, movie_title, pred[ranking[i]] * 5))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("person_id", type=int)
    args = parser.parse_args()
    person_id = args.person_id

    get_ranking_DL(person_id)

    get_ranking_RF(person_id)
