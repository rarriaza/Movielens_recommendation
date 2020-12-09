import pickle

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from pathlib import Path
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

from data_preprocessing import preprocessing


def create_model(input_size=32, n_labels=1):
    model = Sequential()
    model.add(Dense(2*input_size, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(input_size, activation='relu'))
    model.add(Dense(n_labels, activation='sigmoid'))

    model.compile(optimizer="adam",
                  loss="mse")
    return model


def train(model, x, y, n_epochs=10, batch_size=32, callbacks=[]):
    history = model.fit(x, y,
                        batch_size=batch_size,
                        validation_split=0.2,
                        epochs=n_epochs,
                        callbacks=callbacks)
    return history


def run_train(batch_size=32, n_epochs=20, n_features=32):
    model = create_model(input_size=n_features)
    models_dir = Path("checkpoints")
    models_dir.mkdir(exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath=models_dir / 'model-{epoch:02d}-{val_loss:.2f}.h5'),
        tf.keras.callbacks.ModelCheckpoint(filepath=models_dir / 'model_weights-{epoch:02d}-{val_loss:.2f}.h5'
                                           , save_weights_only=True)
    ]

    history = train(model,
                    xtrain, ytrain,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    callbacks=callbacks)
    return model, history


if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest, movies_df, user_df, reviewed_movies = preprocessing()

    # deep learning
    model, history = run_train(batch_size=64, n_epochs=50, n_features=xtrain.shape[1])

    # random forest
    regr = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=300)
    regr.fit(xtrain, ytrain)
    pickle.dump(regr, open("checkpoints/rf_model.pkl", 'wb'))
