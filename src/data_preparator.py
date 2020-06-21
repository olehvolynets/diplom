import pathlib
import pandas as pd

class DataPreparator:
    def __init__(self, path, col_names, sep, na_val, comment, result_col):
        self.path = path             # Шлях до файлу
        self.col_names = col_names   # Назви стовпчиків
        self.sep = sep               # Роздільний символ
        self.na_val = na_val         # Роздільний символ
        self.comment = comment       # Коментар
        self.result_col = result_col # Стовпчик результату

        # Зчитані з таблиці дані
        self.raw_data = pd.read_csv(path, names=col_names,
                                    na_values=na_val, comment=comment,
                                    sep=sep, skipinitialspace=True)

        self.dataset = self.raw_data.copy().dropna()

    def get_dummies(self):
        self.dataset = pd.get_dummies(self.dataset, prefix="", prefix_sep="")
