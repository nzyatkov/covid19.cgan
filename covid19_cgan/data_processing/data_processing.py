class DataProcessing:

    @staticmethod
    def fill_empty_values(data, columns):
        for column in columns:
            if data[column].isnull().values.any():
                # 1. Проводим интерполяцию для значений в середине ряда, если они nan
                data[column] = data[column].interpolate(method='polynomial', order=1)

                # 2. Заполняем первые и последние значения ряда соседними, если они nan
                data[column] = data[column].ffill()
                data[column] = data[column].bfill()

        return data