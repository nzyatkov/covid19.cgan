import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from ta.trend import SMAIndicator
import statsmodels.tsa.api as tsa
from datetime import timedelta
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from data_processing.data_processing import DataProcessing

# Путь до файла для сохранения подготовленного датасета
save_data_file = './data/covid_ml_data_Spb.csv'

# Путь до исходных данных: COVID-19 данные по миру
world_data_file = './data/world/owid-covid-data.csv'
# Путь до исходных данных: COVID-19 данные по Спб
region_data_file = './data/spb/SPb.COVID-19.united.csv'
# Данные IgG Инвитро по Спб
invitro_file = './data/spb/spb-invitro.csv'
# Данные Индекса самоизоляции от Яндекса в Спб
yandex_index_file = './data/spb/spb_data.csv'

# Признаки, которые будем использовать
proc_features = ['new_diagnoses', 'hospitalized', 'new_deaths', 'new_tests', 'new_cases_world_minus_china',
                 'yandex_index', 'IgG']


world_data_data = pd.read_csv(world_data_file, index_col="date", parse_dates=True, na_values=['nan'])
world_data_data = world_data_data.rename(columns={'new_cases_smoothed': 'new_cases_world_minus_china'})

world_data_data = (world_data_data[world_data_data.location == 'World'][['new_cases_world_minus_china']] -
                   world_data_data[world_data_data.location == 'China'][['new_cases_world_minus_china']])

world_data_data.index.names = ['Date']

region_data_data = pd.read_csv(region_data_file, index_col="DATE.spb", parse_dates=True, na_values=['nan']).sort_index()
region_data_data.index.names = ['Date']
region_data_data['AMBUL'] = region_data_data['ACTIVE.spb'] - region_data_data['OCCUPIED_BEDS_CALCULATED']
region_data_data = region_data_data[['AMBUL', 'CONFIRMED.sk', 'DEATHS.sk',
                                     'ACTIVE.spb', 'OCCUPIED_BEDS_CALCULATED',
                                     'VENT', 'PCR_TESTS']].ffill()
region_data_data = region_data_data.rename(columns={'AMBUL': 'ambulatory_cases',
                                                    'CONFIRMED.sk': 'new_diagnoses',
                                                    'DEATHS.sk': 'new_deaths',
                                                    'ACTIVE.spb': 'active_cases',
                                                    'OCCUPIED_BEDS_CALCULATED': 'hospitalized',
                                                    'VENT': 'ventilation',
                                                    'PCR_TESTS': 'new_tests'})


# Invitro Data
invitro_data = pd.read_csv(invitro_file, index_col="date", parse_dates=True, na_values=['nan'])
invitro_data = invitro_data.rename(columns={"positive_percent": "IgG"})
invitro = {}
invitro['Date'] = pd.date_range(invitro_data.index[0], invitro_data.index[-1])
invitro = pd.DataFrame(invitro).set_index('Date')
invitro = invitro.join(invitro_data)

# Yandex Index Data
yandex_index = pd.read_csv(yandex_index_file, index_col="Date", parse_dates=True, na_values=['nan'])[['yandex_index']]

# Объединяем данные в 1 датасет region_data
region_data = {}
region_data['Date'] = pd.date_range(region_data_data.index[0], region_data_data.index[-1])
region_data = pd.DataFrame(region_data).set_index('Date')
region_data = region_data.join(region_data_data)
region_data = region_data.join(invitro)
region_data = region_data.join(yandex_index)
region_data = region_data.join(world_data_data.new_cases_world_minus_china)

# Экстраполируем индекс самоизоляции Яндекса
date_ya_start = region_data['yandex_index'].dropna().index[-1]
res_date = date_ya_start
while res_date <= region_data.index[-1]:
    region_data.loc[res_date, 'yandex_index'] = region_data.loc[res_date - timedelta(days=7), 'yandex_index']
    res_date += timedelta(days=1)

# Экстраполируем данные Инвитро IgG
date_inv_start = region_data['IgG'].dropna().index[-1]
res_date = date_inv_start
while res_date <= region_data.index[-1]:
    region_data.loc[res_date, 'IgG'] = region_data.loc[res_date - timedelta(days=7), 'IgG']
    res_date += timedelta(days=1)

# Добавляем отношение выявленных случаев к количеству проведённых тестов
region_data.loc['2023-07-29', 'new_tests'] = None
region_data['new_diagnoses__new_tests'] = (region_data['new_diagnoses'] / region_data['new_tests'])

# Удаляем выбросы из данных
for date in ['2024-01-04', '2023-01-02', '2023-12-04', '2021-01-04']:
    region_data.loc[date, 'new_diagnoses__new_tests'] = None

# Удаляем выбросы из данных
for date in ['2020-12-07', '2020-11-30', '2020-11-23', '2020-11-16', '2020-11-02', '2020-10-26', '2020-10-19',
             '2020-10-12', '2020-09-28', '2020-09-21', '2020-09-14', '2020-09-07', '2020-08-31', '2020-08-24']:
    region_data.loc[date, 'new_deaths'] = None

# Обрабатываем первичные признаки
features = proc_features.copy()
features.append('new_diagnoses__new_tests')
data = region_data[features]

# Заполняем NaN ячейки путём линейной интерполяции
data = DataProcessing.fill_empty_values(data, features)

# Вторичные признаки
# Сглаживание индикаторов скользящей средней
ema_window = 14
for feature in ['IgG', 'new_deaths', 'new_diagnoses__new_tests']:
    # Сглаживаем индикатор скользящей средней
    transformed_feature = SMAIndicator(close=data[feature], window=ema_window).sma_indicator()

    # Добавляем признак в data
    data[f'{feature}_ema'] = transformed_feature

ema_window = 7
for feature in ['new_diagnoses', 'new_tests']:
    # Сглаживаем индикатор скользящей средней
    transformed_feature = SMAIndicator(close=data[feature], window=ema_window).sma_indicator()

    # Добавляем признак в data
    data[f'{feature}_ema'] = transformed_feature

# lag-дневное приращение логарифма c отсечением выбросов
outlier_cutoff = 0.01
# 3-49-дневные приращения логарифмов
for lag in [3, 7, 10, 14, 21, 28, 35, 42, 49]:

    for feature in ['new_diagnoses_ema',
                    'hospitalized',
                    'new_deaths_ema',
                    'new_tests_ema',
                    'new_cases_world_minus_china',
                    'new_diagnoses__new_tests_ema',
                    'new_diagnoses', 'new_deaths', 'new_tests', 'new_diagnoses__new_tests']:
        # Добавляем 1 к данным, чтобы избежать взятия логарифма от 0
        log_feature = np.log(1 + data[feature])

        # Вычисляем логарифмическое приращение признака
        log_feature = log_feature - log_feature.shift(lag)

        # Отсекаем выбросы нового признака за 1-ым и 99-ым перцентилями
        # log_feature = log_feature.pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
        #                                                 upper=x.quantile(1-outlier_cutoff)))

        # Добавляем признак в data
        data[f'{feature}_log_{lag}d'] = log_feature

# Берём процентное изменение индикатора за lag последних дней
for lag in [3, 7, 10, 14, 21, 28, 35, 42, 49]:
    for feature in ['IgG_ema', 'IgG']:

        # Берём процентное изменение индикатора
        transformed_feature = data[feature].pct_change(lag)

        # Добавляем признак в data
        data[f'{feature}_pc_{lag}d'] = transformed_feature

# Приращение логарифмов новых выявленных случаев к скользящей средней за 7 последних дней
# Сначала добавляем признаки без смещения -t
window = 7
targets = ['new_diagnoses_ema', 'hospitalized', 'new_deaths_ema', 'new_tests_ema', 'new_cases_world_minus_china', 'new_diagnoses__new_tests_ema',
           'new_diagnoses', 'new_deaths', 'new_tests', 'new_diagnoses__new_tests']

for target in targets:
    for t in [3, 7, 10, 14, 21, 28, 35, 42, 49]:

        # Вычисляем SMA за window последних дней для t дней назад
        target_sma = SMAIndicator(close=data[target], window=window).sma_indicator().shift(t)

        # Берём log-разницу между текущим значением минус средним t дней назад
        data[f'{target}_log_diff_sma{window}_ch_{t}d'] = (np.log(data[target] + 1) - np.log(target_sma + 1))

# Добавляем сами таргеты
targets = ['new_diagnoses', 'hospitalized']
for column in targets:
    # Берём трендовую состовляющую временного ряда методом TSA, как способ сгладить наш таргет
    # На деле, по сути, это взятие центрированной скользящей средней
    components = tsa.seasonal_decompose(data[column], model='additive', period=14, extrapolate_trend='freq')
    data[column + '_tsa'] = components.trend

window = 7
for target in ['new_diagnoses_tsa', 'hospitalized_tsa']:
    for t in range(1,15):

        # Вычисляем SMA за window последних дней для t дней назад
        target_sma = SMAIndicator(close=data[target], window=window).sma_indicator().shift(t)

        # Берём log-разницу между текущим значением минус средним t дней назад
        data[f'target_{target}_log_diff_sma{window}_ch_{t}d'] = (np.log(data[target] + 1) - np.log(target_sma + 1)).shift(-t)

for target in ['new_diagnoses_tsa', 'hospitalized_tsa']:
    for t in range(1,15):
        # Берём таргет без изменения целевой функции
        data[f'target_{target}_{t}d'] = data[target].shift(t)

# Сохраняем подготовленный датасет
pd.concat([data[:-14].dropna(), data[-14:]]).to_csv(save_data_file)