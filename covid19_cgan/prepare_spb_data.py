import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from ta.trend import SMAIndicator
import statsmodels.tsa.api as tsa
from datetime import timedelta
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from data_processing.data_processing import DataProcessing

def run_prepare_spb_data():
    # Path to the file to save the prepared dataset
    save_data_file = './data/covid_ml_data_Spb.csv'

    # The Path to the Raw Data: COVID-19 Data Around the World
    world_data_file = './data/world/owid-covid-data.csv'
    # Path to the original data: COVID-19 data for St. Petersburg
    region_data_file = './data/spb/SPb.COVID-19.united.csv'
    # IgG Invitro data for St. Petersburg
    invitro_file = './data/spb/spb-invitro.csv'
    # Yandex Self-Isolation Index data in St. Petersburg
    yandex_index_file = './data/spb/spb_data.csv'

    # Features we will use
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

    # Combine data into 1 dataset region_data
    region_data = {}
    region_data['Date'] = pd.date_range(region_data_data.index[0], region_data_data.index[-1])
    region_data = pd.DataFrame(region_data).set_index('Date')
    region_data = region_data.join(region_data_data)
    region_data = region_data.join(invitro)
    region_data = region_data.join(yandex_index)
    region_data = region_data.join(world_data_data.new_cases_world_minus_china)

    # Extrapolating Yandex's self-isolation index
    date_ya_start = region_data['yandex_index'].dropna().index[-1]
    res_date = date_ya_start
    while res_date <= region_data.index[-1]:
        region_data.loc[res_date, 'yandex_index'] = region_data.loc[res_date - timedelta(days=7), 'yandex_index']
        res_date += timedelta(days=1)

    # Extrapolating Invitro IgG data
    date_inv_start = region_data['IgG'].dropna().index[-1]
    res_date = date_inv_start
    while res_date <= region_data.index[-1]:
        region_data.loc[res_date, 'IgG'] = region_data.loc[res_date - timedelta(days=7), 'IgG']
        res_date += timedelta(days=1)

    # Add the ratio of detected cases to the number of tests performed
    region_data.loc['2023-07-29', 'new_tests'] = None
    region_data['new_diagnoses__new_tests'] = (region_data['new_diagnoses'] / region_data['new_tests'])

    # Removing outliers from data
    for date in ['2024-01-04', '2023-01-02', '2023-12-04', '2021-01-04']:
        region_data.loc[date, 'new_diagnoses__new_tests'] = None

    # Removing outliers from data
    for date in ['2020-12-07', '2020-11-30', '2020-11-23', '2020-11-16', '2020-11-02', '2020-10-26', '2020-10-19',
                 '2020-10-12', '2020-09-28', '2020-09-21', '2020-09-14', '2020-09-07', '2020-08-31', '2020-08-24']:
        region_data.loc[date, 'new_deaths'] = None

    # Processing primary features
    features = proc_features.copy()
    features.append('new_diagnoses__new_tests')
    data = region_data[features]

    # Fill NaN cells using linear interpolation
    data = DataProcessing.fill_empty_values(data, features)

    # Secondary features
    # Smoothing by Moving Average
    ema_window = 14
    for feature in ['IgG', 'new_deaths', 'new_diagnoses__new_tests']:
        # Smoothing with a moving average indicator
        transformed_feature = SMAIndicator(close=data[feature], window=ema_window).sma_indicator()

        # Add a feature to data
        data[f'{feature}_ema'] = transformed_feature

    ema_window = 7
    for feature in ['new_diagnoses', 'new_tests']:
        # Smoothing with a moving average indicator
        transformed_feature = SMAIndicator(close=data[feature], window=ema_window).sma_indicator()

        # Add a feature to data
        data[f'{feature}_ema'] = transformed_feature

    # lag-day logarithm increment with outlier trimming
    outlier_cutoff = 0.01
    # 3-49 day log increments
    for lag in [3, 7, 10, 14, 21, 28, 35, 42, 49]:

        for feature in ['new_diagnoses_ema',
                        'hospitalized',
                        'new_deaths_ema',
                        'new_tests_ema',
                        'new_cases_world_minus_china',
                        'new_diagnoses__new_tests_ema',
                        'new_diagnoses', 'new_deaths', 'new_tests', 'new_diagnoses__new_tests']:
            # Add 1 to the data to avoid taking the logarithm of 0
            log_feature = np.log(1 + data[feature])

            # Calculate the logarithmic increment of the feature
            log_feature = log_feature - log_feature.shift(lag)

            # Cut off outliers of the new feature at the 1st and 99th percentiles
            # log_feature = log_feature.pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
            #                                                 upper=x.quantile(1-outlier_cutoff)))

            # Add a feature to data
            data[f'{feature}_log_{lag}d'] = log_feature

    # The percentage change of the indicator for the last days lag
    for lag in [3, 7, 10, 14, 21, 28, 35, 42, 49]:
        for feature in ['IgG_ema', 'IgG']:

            # Берём процентное изменение индикатора
            transformed_feature = data[feature].pct_change(lag)

            # Добавляем признак в data
            data[f'{feature}_pc_{lag}d'] = transformed_feature

    # Increment of logarithms of new cases to the moving average for the last 7 days
    # First, add features without offset -t
    window = 7
    targets = ['new_diagnoses_ema', 'hospitalized', 'new_deaths_ema', 'new_tests_ema', 'new_cases_world_minus_china', 'new_diagnoses__new_tests_ema',
               'new_diagnoses', 'new_deaths', 'new_tests', 'new_diagnoses__new_tests']

    for target in targets:
        for t in [3, 7, 10, 14, 21, 28, 35, 42, 49]:

            # Calculate SMA for the last days window for t days ago
            target_sma = SMAIndicator(close=data[target], window=window).sma_indicator().shift(t)

            # Take the log difference between the current value minus the average t days ago
            data[f'{target}_log_diff_sma{window}_ch_{t}d'] = (np.log(data[target] + 1) - np.log(target_sma + 1))

    # Add the targets
    targets = ['new_diagnoses', 'hospitalized']
    for column in targets:
        # Take the trend component of the time series using the TSA method as a way to smooth out our target
        # In fact, this is essentially a centered moving average
        components = tsa.seasonal_decompose(data[column], model='additive', period=14, extrapolate_trend='freq')
        data[column + '_tsa'] = components.trend

    window = 7
    for target in ['new_diagnoses_tsa', 'hospitalized_tsa']:
        for t in range(1,15):

            # Calculate SMA for the last days window for t days ago
            target_sma = SMAIndicator(close=data[target], window=window).sma_indicator().shift(t)

            # Take the log difference between the current value minus the average t days ago
            data[f'target_{target}_log_diff_sma{window}_ch_{t}d'] = (np.log(data[target] + 1) - np.log(target_sma + 1)).shift(-t)

    for target in ['new_diagnoses_tsa', 'hospitalized_tsa']:
        for t in range(1,15):
            # Take the target without changing the target function
            data[f'target_{target}_{t}d'] = data[target].shift(t)

    # Save the prepared dataset
    pd.concat([data[:-14].dropna(), data[-14:]]).to_csv(save_data_file)

if __name__ == "__main__":
    run_prepare_spb_data()