
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling

class PricePredictor:
    def __init__(self, data, epochs=1000):
        self.data = data     # екземпляр класу DataPreparator
        self.epochs = epochs # кількість епох
        # фрагмент даних, що використовується для тренування
        self.train_set = data.dataset.sample(frac=0.8, random_state=0)
        # фрагмент даних, для тестування
        self.test_set = data.dataset.drop(self.train_set.index)
        self.train_stats = self.train_set.describe().pop(data.result_col).transpose()
        # значення для перевірки тренування
        self.train_labels = self.train_set.pop(data.result_col)
        # значення для перевірки тестування
        self.test_labels = self.test_set.pop(data.result_col)
        # приведення даних до спільниго виду, необхідно для роботи системи
        self.normed_train_data = self.norm(self.train_set)
        self.normed_test_data = self.norm(self.test_set)
        # модель нейронної мережі
        self.model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=[len(self.train_set.keys())]),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        self.model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(0.001), metrics=["mae", "mse"])

    def norm(self, x):
        return (x - self.train_stats["mean"]) / self.train_stats["std"]

    def history(self):
        return self.model.fit(self.normed_train_data, self.train_labels,
                              epochs=self.epochs, validation_split=0.2,
                              verbose=0, callbacks=[tfdocs.modeling.EpochDots()])

    def evaluate(self):
        return self.model.evaluate(self.normed_test_data, self.test_labels, verbose=2)

    def predict(self):
        return self.model.predict(self.normed_test_data).flatten()
