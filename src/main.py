import pathlib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from price_predictor import PricePredictor
from data_preparator import DataPreparator
import pandas as pd

from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots

dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# Назви колонок у CSV таблиці
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin"
]
separator = " "
na_value = "?"
comment = "\t"
result_column_name = "MPG"

def _smooth(values, std):
    width = std * 4
    x = np.linspace(-width, width, 2 * width + 1)
    kernel = np.exp(-(x / 5)**2)

    values = np.array(values)
    weights = np.ones_like(values)

    smoothed_values = np.convolve(values, kernel, mode='same')
    smoothed_weights = np.convolve(weights, kernel, mode='same')

    return smoothed_values / smoothed_weights

data = DataPreparator(path = dataset_path,
                      col_names = column_names,
                      sep=separator,
                      na_val=na_value,
                      comment=comment,
                      result_col=result_column_name)

data.dataset["Origin"] = data.dataset["Origin"].map({
    1: "USA",
    2: "Europe",
    3: "Japan"
})
data.get_dummies()

predictor = PricePredictor(data)

history = predictor.history()
loss, mae, mse = predictor.evaluate()
test_predictions = predictor.predict()
errors = test_predictions - predictor.test_labels

train_val = _smooth(history.history['mae'], std=2)
value = _smooth(history.history['val_mae'], std=2)

fig, (axs1, axs2, axs3) = plt.subplots(1, 3)

axs1.plot(history.epoch,
          train_val,
          label="Помилка %")
axs1.plot(history.epoch,
          value,
          "--",
          label="Помилка значення")
axs1.set(xlabel="Епоха", ylabel="Абсолютне значення помилки")
axs1.legend()

axs2.hist(errors, bins=25)
axs2.set(xlabel="Помилка передбачення", ylabel="Кількість помилок")

axs3.scatter(predictor.test_labels, test_predictions)
axs3.set(xlabel="Значення", ylabel="Передбачення")
axs3.set(xlim=[0, 50], ylim=[0, 50])
axs3.plot([0, 50], [0, 50])

plt.show()
