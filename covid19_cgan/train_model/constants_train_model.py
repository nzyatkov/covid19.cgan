from datetime import datetime

covid_ml_data_file = "../data/covid_ml_data_Spb.csv"

batch_size = 20
epochs = 200
gen_lr = 1e-4
discr_lr = 1e-4

noise_dim = 100

nsample = 14
nsample_forward = 5

# start of training
start_train = datetime(2020, 9, 15)
# end of training
split_date = datetime(2022, 7, 1)

sma_window = 7
targets = ['new_diagnoses_tsa']

selected_features = [
    'new_diagnoses_ema',
    'new_cases_world_minus_china',
    'new_diagnoses__new_tests_ema',

    'new_diagnoses_ema_log_3d',
    'new_diagnoses_ema_log_7d',
    'new_diagnoses_ema_log_10d',

    'new_diagnoses__new_tests_ema_log_3d',
    'new_diagnoses__new_tests_ema_log_7d',
    'new_diagnoses__new_tests_ema_log_10d',

    'hospitalized_log_3d',
    'hospitalized_log_7d',
    'hospitalized_log_10d',

    'new_deaths_ema_log_21d',

    'new_cases_world_minus_china_log_7d',
    'new_cases_world_minus_china_log_14d',
    'new_cases_world_minus_china_log_21d',
    'new_cases_world_minus_china_log_28d',
    'new_cases_world_minus_china_log_42d',

    'IgG_ema',

    'yandex_index'
]

features_to_normalize = [
    'new_diagnoses_ema',
    'new_cases_world_minus_china',
    'new_diagnoses__new_tests_ema'
]

gen_params = {
    'nsample': nsample,
    'n_features': len(selected_features),
    'noise_dim': noise_dim,
    'lstm_units_1': 32,
    'lstm_units_2': 15,
    'dense_units': 32,
    'output_size': nsample_forward
}

discr_params = {
    'nsample': nsample,
    'n_features': len(selected_features),
    'nsample_forward': nsample_forward,
    'lstm_units_1': 32,
    'lstm_units_2': 15,
    'dense_units': 10
}