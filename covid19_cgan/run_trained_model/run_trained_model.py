import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import tensorflow as tf
from keras.models import load_model
from pathlib import Path
import plotly.graph_objects as go
from ta.trend import SMAIndicator

import constants_run_trained_model as CONSTANTS
from data_processing.spb_data_processing import SpbDataProcessing
from logger.logger import get_logger
from utils.common import normalize_noise

def run_trained_model():

    _logger = get_logger()

    # Parameters
    covid_ml_data_file = CONSTANTS.covid_ml_data_file
    NTraj = CONSTANTS.NTraj
    nsample = CONSTANTS.nsample
    nsample_forward = CONSTANTS.nsample_forward
    start_train = CONSTANTS.start_train
    split_date = CONSTANTS.split_date
    sma_window = CONSTANTS.sma_window
    targets = CONSTANTS.targets
    selected_features = CONSTANTS.selected_features
    features_to_normalize = CONSTANTS.features_to_normalize
    noise_dim = CONSTANTS.noise_dim

    targets_days = [d for d in range(1, nsample_forward + 1)]
    target_vars = [None] * len(targets)
    for i in range(len(targets)):
        target_vars[i] = [f'target_{targets[i]}_log_diff_sma{sma_window}_ch_{t}d' for t in targets_days]

    target_vars_flatten = list(np.array(target_vars).reshape(-1))

    # Choosing a device for computing
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        _logger.info('Using GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        _logger.info('Using CPU')

    # Preparing data in RNN format (for the generator)
    X, y = SpbDataProcessing.prepare_spb_rnn_dataset(covid_ml_data_file, selected_features, features_to_normalize,
                                                     target_vars, target_vars_flatten,
                                                     start_train, nsample, nsample_forward)

    # Load model
    model_path = Path('../data', 'model')
    generator = load_model(model_path / CONSTANTS.generator_model_name)
    discriminator = load_model(model_path / CONSTANTS.discriminator_model_name)

    # =============================================================================
    # Forecast NTraj times on data
    all_y = [[]] * len(targets)
    all_y_predicted = [[]] * len(targets)

    for ntraj in range(NTraj):
        noise = normalize_noise(tf.random.normal([X.shape[0], noise_dim]))
        y_predicted = generator.predict([np.stack(X.values), noise], verbose=0)

        _logger.info(f'{ntraj+1} forecast trajectory from {NTraj} done, shape = {y_predicted.shape}')

        all_y_predicted[0].append(y_predicted)

    all_y_predicted[0] = np.array(all_y_predicted[0])
    all_y[0] = pd.DataFrame(data=np.stack(y.values), index=y.index, columns=target_vars[0])

    # =============================================================================
    # Plotting true and pred graphs
    all_y_predicted_plot = {}
    all_y_predicted_std_plot = {}
    all_y_predicted_plot[0] = pd.DataFrame(data=np.mean(all_y_predicted[0], axis=0),
                                           index=y.index, columns=target_vars[0])
    all_y_predicted_std_plot[0] = pd.DataFrame(data=np.std(all_y_predicted[0], axis=0),
                                               index=y.index, columns=target_vars[0])
    for j in range(len(targets)):
        for i, target_var in enumerate(target_vars[j]):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=all_y[j].index,
                                     y=all_y[j][target_var],
                                     name=f'true mean F(t+{i+1})'))
            fig.add_trace(go.Scatter(x=all_y_predicted_plot[j].index,
                                     y=all_y_predicted_plot[j][target_var],
                                     name=f'pred mean F(t+{i+1})'))

            all_min = all_y_predicted_plot[j][target_var] - 3 * all_y_predicted_std_plot[j][target_var]
            all_max = all_y_predicted_plot[j][target_var] + 3 * all_y_predicted_std_plot[j][target_var]
            fig.add_trace(go.Scatter(
                x=np.concatenate([all_min.index, all_max.index[::-1]]),
                y=pd.concat([all_min, all_max[::-1]]),
                fill='toself',
                mode='lines',
                line_color='rgba(229,90,66,0.2)',
                fillcolor='rgba(229,90,66,0.2)',
                name="$+/-3 \sigma$ conf interval"
            ))

            fig.update_layout(template='plotly_white',
                              title=f'{targets[j]}, days forward = {targets_days[target_vars[j].index(target_var)]}')
            fig.write_image(f'./data/results/true_vs_pred_{NTraj}_F(t+{i + 1}).png',
                            width=1000,
                            height=500,
                            scale=1,
                            format="png")
            _logger.info(f'true_vs_pred_{NTraj}_F(t+{i + 1}).png successfully saved')

    # =============================================================================
    # Inverse transformation MinMaxScaler

    _logger.info('Start MinMaxScaler inverse transform...')

    y_pred_transformed = SpbDataProcessing.pred_inverse_transform(targets, target_vars, all_y, all_y_predicted, NTraj)

    _logger.info('MinMaxScaler inverse transform done.')

    # =============================================================================
    # Recovering real forecasts from stationary targets

    _logger.info('Start real forecasts f transform from targets F...')

    forecast = SpbDataProcessing.transform_from_stationary(y_pred_transformed, targets, targets_days,
                                                           sma_window, nsample_forward, NTraj)

    _logger.info('Real forecasts f transform from targets F done.')

    # =============================================================================
    # Plot real data and forecasts

    forecast_std = [None] * len(targets)
    forecast_std[0] = pd.DataFrame(data=np.std(forecast[0], axis=0),
                                   index=y.index, columns=[f'{targets[0]}_{j}d' for j in targets_days])
    forecast[0] = pd.DataFrame(data=np.mean(np.array(forecast[0]), axis=0),
                               index=y.index, columns=[f'{targets[0]}_{j}d' for j in targets_days])


    for i in range(len(targets)):

        new_diagnoses_addition = pd.DataFrame(index=pd.date_range(start=forecast[i].index[-1] + timedelta(days=1),
                                                                  end=forecast[i].index[-1] + timedelta(days=45)),
                                              #columns=forecast[i].columns)
                                              columns=[f'{targets[i]}_{j}d' for j in targets_days])

        forecast_all = pd.concat([forecast[i], new_diagnoses_addition])
        forecast_std_all = pd.concat([forecast_std[i], new_diagnoses_addition])


        for j, target_var in enumerate(targets_days):

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=forecast_all[f'{targets[i]}_{target_var}d'].shift(target_var).index,
                                     y=forecast_all[f'{targets[i]}_{target_var}d'].shift(target_var),
                                     name=f'{targets[i]}_forecast_{target_var}d'))

            all_min = (forecast_all[f'{targets[i]}_{target_var}d'] - 3*forecast_std_all[f'{targets[i]}_{target_var}d']).shift(target_var)
            all_max = (forecast_all[f'{targets[i]}_{target_var}d'] + 3*forecast_std_all[f'{targets[i]}_{target_var}d']).shift(target_var)
            fig.add_trace(go.Scatter(
                x=np.concatenate([all_min.index, all_max.index[::-1]]),
                y=pd.concat([all_min, all_max[::-1]]),
                fill='toself',
                mode='lines',
                line_color='rgba(99,110,250,0.3)',
                fillcolor='rgba(99,110,250,0.3)',
                name='pred_target min и max'
            ))

            fig.add_trace(go.Scatter(x=SpbDataProcessing.data[targets[i]].index,
                                     y=SpbDataProcessing.data[targets[i]],
                                     line_color='rgba(229,90,66,0.8)',
                                     name=targets[i]))

            fig.update_layout(template='plotly_white',
                              title=f'Прогноз {targets[i]} случаев по Санкт-Петербургу на {target_var} дней с {y.index[0]}',
                              legend_orientation="v",
                              legend=dict(x=0.5, xanchor="center"))
            fig.write_image(f'./data/results/new_cases_f(t+{j + 1}).png',
                            width=1000,
                            height=500,
                            scale=1,
                            format="png")

            _logger.info(f'new_cases_f(t+{j + 1}).png successfully saved')
