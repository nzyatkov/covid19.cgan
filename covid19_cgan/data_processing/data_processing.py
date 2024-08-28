import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataProcessing:
    @staticmethod
    def fill_empty_values(data, columns):
        for column in columns:
            if data[column].isnull().values.any():
                # 1. Interpolation for values of the series if they are nan
                data[column] = data[column].interpolate(method='polynomial', order=1)

                # 2. Fill the first and last values of the series with the neighboring ones if they are nan
                data[column] = data[column].ffill()
                data[column] = data[column].bfill()

        return data
