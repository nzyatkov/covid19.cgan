import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU, Reshape, Input, GRU, Dropout, ReLU, concatenate, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy
from pathlib import Path

# import matplotlib.pyplot as plt
import plotly.graph_objects as go

import seaborn as sns

# from time import time

from ta.trend import EMAIndicator, SMAIndicator

import constants_run_model as CONSTANTS
from logger.logger import get_logger

_logger = get_logger()

def prepare_data(X_residual_features, X_features_to_normalize, basis_Y, look_back):
    datares = []
    for start in range(X_residual_features.shape[0] - look_back + 1):
        res_part = X_residual_features[start:start + look_back]

        if features_to_normalize != []:
            norm_part = X_features_to_normalize[start:start + look_back]
            norm_part = pd.DataFrame(data=MinMaxScaler().fit_transform(norm_part),
                                     index=norm_part.index,
                                     columns=norm_part.columns)

            res_part = res_part.join(norm_part)

        datares.append(res_part.values)

    X = np.atleast_3d(np.array(datares))
    X = pd.Series(data=list(X),
                  index=X_residual_features.index[look_back - 1:])

    y = basis_Y[look_back - 1:]

    y = pd.Series(data=list(y.values),
                  index=y.index)

    return X, y

def normalize_noise(noise):
    return tf.divide(
               tf.subtract(
                  noise,
                  tf.reduce_min(noise)
               ),
               tf.subtract(
                  tf.reduce_max(noise),
                  tf.reduce_min(noise)
               )
           )

# Параметры
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

# Выбираем устройство для вычислений
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    _logger.info('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    _logger.info('Using CPU')

# Load Data
data = pd.read_csv(covid_ml_data_file, index_col="Date", parse_dates=True, na_values=['nan'])

# Готовим данные для обучения
residual_features = [item for item in selected_features if item not in features_to_normalize]
ml_data = data[selected_features + target_vars_flatten]
ml_data = ml_data.loc[start_train:]
ml_data = ml_data[:-nsample_forward]

# Нормализация данных
mm_scaler = MinMaxScaler()
ml_data_transformed = pd.DataFrame(data=mm_scaler.fit_transform(ml_data[residual_features + target_vars_flatten]),
                                   index=ml_data.index,
                                   columns=ml_data[residual_features + target_vars_flatten].columns)
ml_data_transformed = ml_data_transformed.join(ml_data[features_to_normalize])


# Готовим пары входных и выходных данных для рекуррентной сети
X, y = prepare_data(ml_data_transformed[residual_features],
                    ml_data_transformed[features_to_normalize],
                    ml_data_transformed[target_vars[0]],
                    nsample)

# Split data on train and test
# train_x, train_y = X[(X.index >= start_train)&(X.index <= split_date)], y[(X.index >= start_train)&(y.index <= split_date)]
# test_x, test_y = X[X.index > split_date], y[y.index > split_date]

# Определяем архитектуру нейронной сети


# Генератор
# lstm_units_1 = 32
# lstm_units_2 = 15
# dense_units = 32
# output_size = nsample_forward
# n_features = len(selected_features)

# def build_generator():
#     # Inputs
#     context = Input(shape=(nsample, n_features),
#                     name='Context')
#     noise = Input(shape=(noise_dim,), name='Noise')
#
#     lstm1 = LSTM(units=lstm_units_1,
#                  input_shape=(nsample, n_features),
#                  name='LSTM1',
#                  dropout=.15,
#                  recurrent_dropout=.1,
#                  return_sequences=True)(context)
#
#     lstm_model = LSTM(units=lstm_units_2,
#                       input_shape=(nsample, n_features),
#                       dropout=.15,
#                       recurrent_dropout=.1,
#                       name='LSTM2')(lstm1)
#
#     # Concat model components
#     merged = concatenate([lstm_model,
#                           noise], name='Merged')
#
#     bn = BatchNormalization(name='bn')(merged)
#
#     hidden_dense = Dense(dense_units, activation='relu', name='FC1')(bn)
#     # relu = ReLU(name='RELU_G')(hidden_dense)
#     # relu = LeakyReLU(alpha=0.2, name='LeakyReLU_G')(hidden_dense)
#
#     output_1 = Dense(output_size, activation='sigmoid', name='new_cases_forecast')(hidden_dense)
#
#     gen = Model(inputs=[context, noise], outputs=output_1, name='Generator')
#
#     return gen
#
# generator = build_generator()
#
# lstm_units_1 = 32
# lstm_units_2 = 15
#
# dense_units = 10
# # output_size = nsample_forward
# n_features = len(selected_features)
#
# def build_discriminator():
#     # Inputs
#     context = Input(shape=(nsample, n_features),
#                     name='Context')
#     forecast = Input(shape=(output_size,), name='Forecast')
#
#     lstm1 = LSTM(units=lstm_units_1,
#                  input_shape=(nsample, n_features),
#                  name='LSTM1',
#                  dropout=.15,
#                  recurrent_dropout=.1,
#                  return_sequences=True)(context)
#
#     lstm_model = LSTM(units=lstm_units_2,
#                       input_shape=(nsample, n_features),
#                       dropout=.15,
#                       recurrent_dropout=.1,
#                       name='LSTM2')(lstm1)
#
#     # Concat model components
#     merged = concatenate([lstm_model,
#                           forecast], name='Merged')
#
#     hidden_dense = Dense(dense_units, activation='relu', name='FC1')(merged)
#     # bn = BatchNormalization(name='bn')(hidden_dense)
#     # relu = ReLU(name='RELU_D')(hidden_dense)
#     # relu = LeakyReLU(alpha=0.2, name='LeakyReLU_D')(hidden_dense)
#
#     # output_size = 1 - на выходе прогнозируем 1 значение - 7-Дневное
#     #                скалированное от 0 до 1 log изменение новых выявленных случаев
#     output_1 = Dense(1, activation='sigmoid', name='Output_D')(hidden_dense)
#
#     dis = Model(inputs=[context, forecast], outputs=output_1, name='Discriminator')
#
#     return dis
#
# discriminator = build_discriminator()

# batch_size = 20
# epochs = 200

# Готовим данные для нейронной сети в формате tf
# train_set = (tf.data.Dataset
#              .from_tensor_slices(({"Context": np.stack(train_x.values)}, {"new_cases_forecast": np.stack(train_y.values)}))
#              .shuffle(buffer_size=train_x.shape[0])
#              .batch(batch_size=batch_size))
#
# test_set = (tf.data.Dataset
#              .from_tensor_slices(({"Context": np.stack(test_x.values)}, {"new_cases_forecast": np.stack(test_y.values)}))
#              .shuffle(buffer_size=test_x.shape[0])
#              .batch(batch_size=batch_size))
#
#
# cross_entropy = BinaryCrossentropy(from_logits=False)

# Generator Loss
# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)

# Discriminator Loss
# def discriminator_loss(true_output, fake_output):
#     true_loss = cross_entropy(tf.ones_like(true_output), true_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     return true_loss + fake_loss

# Optimizers
# gen_optimizer = Adam(1e-4)
# dis_optimizer = Adam(1e-4)

# Проверяем генератор и дискриминатор
# noise = normalize_noise(tf.random.normal([1, noise_dim]))
# generated_seq = generator([tf.convert_to_tensor(train_x.iloc[0].reshape(1,14,20)), noise], training=False)
# print(generated_seq)
# print(discriminator([tf.convert_to_tensor(train_x.iloc[0].reshape(1,14,20)), generated_seq]).numpy())

# Загрузить модель
model_path = Path('data', 'model')
generator = load_model(model_path / CONSTANTS.generator_model_name)
discriminator = load_model(model_path / CONSTANTS.discriminator_model_name)


# # Прогнозируем F(t+1) - F(t+5) на всём периоде
# all_y = [None]*len(targets)
# all_y_predicted = [None]*len(targets)
#
# noise = normalize_noise(tf.random.normal([X.shape[0], noise_dim]))
# y_predicted = generator.predict([np.stack(X.values), noise])
#
# print(f'y_predicted.shape =', y_predicted.shape)
#
# all_y[0] = pd.DataFrame(data=np.stack(y.values), index=y.index)
# all_y_predicted[0] = pd.DataFrame(data=y_predicted, index=y.index)
#
# for j in range(len(targets)):
#     for i in range(nsample_forward):
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=all_y[j].index,
#                                  y=all_y[j][i],
#                                  name=f'true F(t+{i+1})'))
#         fig.add_trace(go.Scatter(x=all_y_predicted[j].index,
#                                  y=all_y_predicted[j][i],
#                                  name=f'pred F(t+{i+1})'))
#         fig.update_layout(template='plotly_white', title=f'{targets[j]}, days forward = {targets_days[i]}, $\sigma$')
#         fig.write_image(f'./data/results/true_vs_pred_F(t+{i+1}).png',
#                         width=1000,
#                         height=500,
#                         scale=1,
#                         format="png")

# =============================================================================

# Прогнозируем NTraj раз
all_y = [[]] * len(targets)
all_y_predicted = [[]] * len(targets)

for ntraj in range(NTraj):
    noise = normalize_noise(tf.random.normal([X.shape[0], noise_dim]))
    y_predicted = generator.predict([np.stack(X.values), noise], verbose=0)

    _logger.info(f'{ntraj+1} forecast trajectory from {NTraj} done, shape = {y_predicted.shape}')

    all_y_predicted[0].append(y_predicted)

all_y_predicted[0] = np.array(all_y_predicted[0])
all_y[0] = pd.DataFrame(data=np.stack(y.values), index=y.index, columns=target_vars[0])

# Строим графики true и pred
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
# Обратное преобразование MinMaxScaler

_logger.info('Start MinMaxScaler inverse transform...')
y_pred_transformed = [None]*len(targets)

for i in range(len(targets)):
    y_pred_transformed[i] = {}

    col = 'true'
    ml_data_transformed_copy = ml_data_transformed[residual_features].loc[all_y[i].index[0]:all_y[i].index[-1]]
    for j in range(len(targets)):
        ml_data_transformed_copy[target_vars[j]] = all_y[j].values
    y_pred_transformed[i][col] = pd.DataFrame(data=mm_scaler.inverse_transform(ml_data_transformed_copy),
                                                    index=ml_data_transformed_copy.index,
                                                    columns=ml_data_transformed_copy.columns)[target_vars[i]]

    col = 'pred'
    y_pred_transformed[i][col] = []
    for ntraj in range(NTraj):
        ml_data_transformed_copy = ml_data_transformed[residual_features].loc[all_y[i].index[0]:all_y[i].index[-1]]
        for j in range(len(targets)):
            ml_data_transformed_copy[target_vars[j]] = all_y_predicted[j][ntraj]
        inv_tr_res_df = pd.DataFrame(data=mm_scaler.inverse_transform(ml_data_transformed_copy),
                                     index=ml_data_transformed_copy.index,
                                     columns=ml_data_transformed_copy.columns)[target_vars[i]]
        y_pred_transformed[i][col].append(inv_tr_res_df)
_logger.info('MinMaxScaler inverse transform done.')

# =============================================================================
# Восстанавливаем реальные прогнозы из таргетов

_logger.info('Start real forecasts f transform from targets F...')
forecast = [None]*len(targets)
forecast_std = [None]*len(targets)

for i in range(len(targets)):
    forecast[i] = {}
    forecast_std[i] = {}

res_type = 'pred'

forecast[0] = []

for i in range(len(targets)):
    for ntraj in range(NTraj):

        frc = pd.DataFrame(data=np.zeros([y_pred_transformed[i][res_type][ntraj].shape[0],
                                               y_pred_transformed[i][res_type][ntraj].shape[1]]),
                           index=y_pred_transformed[i][res_type][ntraj].index,
                           columns=[f'{targets[i]}_{j}d' for j in targets_days])

        target_sma = SMAIndicator(close=data[targets[i]], window=sma_window).sma_indicator()

        for idx in range(y_pred_transformed[i][res_type][ntraj].shape[0]):

            for target_var in range(nsample_forward):
                frc.iloc[idx, target_var] = np.exp(

                    y_pred_transformed[i][res_type][ntraj].iloc[idx, target_var] +
                    np.log(target_sma.loc[y_pred_transformed[i][res_type][ntraj].index[idx]] + 1)

                ) - 1

        forecast[i].append(frc.values)

        _logger.info(f'{ntraj+1} from {NTraj} forecast done.')

_logger.info('Real forecasts f transform from targets F done.')

# =============================================================================
# Отрисовываем реальные данные и прогнозы

forecast[0] = np.array(forecast[0])
forecast_std[0] = pd.DataFrame(data=np.std(forecast[0], axis=0),
                               index=y.index, columns=[f'{targets[0]}_{j}d' for j in targets_days])
forecast[0] = pd.DataFrame(data=np.mean(forecast[0], axis=0),
                           index=y.index, columns=[f'{targets[0]}_{j}d' for j in targets_days])


region_data_data = data

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

        fig.add_trace(go.Scatter(x=region_data_data[targets[i]].index,
                                 y=region_data_data[targets[i]],
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