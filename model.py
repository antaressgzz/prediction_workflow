import hashlib
import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

tf.random.set_seed(2021)


class Model:
    def __init__(self):
        self.id = self._hash_id() # 生成模型 ID
        self.data = Preprocessor()

    @staticmethod
    def _hash_id():
        return hashlib.md5(open('model.py', 'rb').read()).hexdigest()

    def fit(self, training_set):
        self.build_nn()
        X_tr, y_tr = training_set
        self.nn.fit(X_tr, y_tr, batch_size=64, epochs=50, verbose=0)

    def predict(self, inputs):
        return self.nn.predict(inputs, batch_size=64)

    def build_nn(self):
        tf.random.set_seed(2021)
        inputs = keras.Input(shape=(2,))
        x = layers.Dense(16, activation="relu")(inputs)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        self.nn = keras.Model(inputs=inputs, outputs=outputs)
        self.nn.compile(
            loss='mse',
            optimizer=keras.optimizers.SGD(learning_rate=5e-4),
            metrics=["mse"],
        )


class Preprocessor:
    def __init__(self):
        self.X = np.random.uniform(-10, 10, (1000, 2))
        self.y = self.X[:, 0:1] ** 2 + self.X[:, 1:2] ** 2 + np.random.normal(0, 1, (1000, 1))
        self.idx_iter = KFold().split(self.X)

    def __iter__(self):
        return self

    def __next__(self):
        train_idx, val_idx = next(self.idx_iter)
        X_train, X_val = self.X[train_idx], self.X[val_idx]
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std
        y_train, y_val = self.y[train_idx], self.y[val_idx]
        return (X_train, y_train), (X_val, y_val)


if __name__ == '__main__':
    model = Model()
    print(model.id)
    it = iter(model.data)
    tr, val = next(it)
    print(tr[0].shape)


