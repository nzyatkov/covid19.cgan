from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from keras.optimizers import Adam
import tensorflow.keras.backend as K

import train_model.constants_train_model as CONSTANTS
from data_processing.spb_data_processing import SpbDataProcessing
from logger.logger import get_logger
from nn_model.cgan import CGAN
from nn_model.discriminator import Discriminator
from nn_model.generator import Generator
from utils.common import normalize_noise

def train_model():
    _logger = get_logger()

    MODEL_PATH = Path('data', 'model')
    if not MODEL_PATH.exists():
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

    RESULTS_PATH = Path('data', 'results')
    if not RESULTS_PATH.exists():
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Parameters
    covid_ml_data_file = CONSTANTS.covid_ml_data_file
    nsample = CONSTANTS.nsample
    nsample_forward = CONSTANTS.nsample_forward
    start_train = CONSTANTS.start_train
    split_date = CONSTANTS.split_date
    sma_window = CONSTANTS.sma_window
    targets = CONSTANTS.targets
    selected_features = CONSTANTS.selected_features
    features_to_normalize = CONSTANTS.features_to_normalize
    noise_dim = CONSTANTS.noise_dim
    batch_size = CONSTANTS.batch_size
    epochs = CONSTANTS.epochs
    gen_lr = CONSTANTS.gen_lr
    discr_lr = CONSTANTS.discr_lr
    gen_params = CONSTANTS.gen_params
    discr_params = CONSTANTS.discr_params

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

    # Preparing data in RNN format
    _logger.info('Preparing data in RNN format...')
    X, y = SpbDataProcessing.prepare_spb_rnn_dataset(covid_ml_data_file, selected_features, features_to_normalize,
                                                     target_vars, target_vars_flatten,
                                                     start_train, nsample, nsample_forward)
    _logger.info('Data in RNN format successfully prepared.')

    # Split data on train and test
    train_x, train_y = (X[(X.index >= start_train) & (X.index <= split_date)],
                        y[(X.index >= start_train) & (y.index <= split_date)])
    test_x, test_y = X[X.index > split_date], y[y.index > split_date]


    K.clear_session()

    # Prepare train and test sets in tensorflow format
    train_set = (tf.data.Dataset
                 .from_tensor_slices((np.stack(train_x.values), np.stack(train_y.values)))
                 .shuffle(buffer_size=train_x.shape[0])
                 .batch(batch_size=batch_size))

    test_set = (tf.data.Dataset
                .from_tensor_slices((np.stack(test_x.values), np.stack(test_y.values)))
                .batch(batch_size=batch_size))

    # Generator
    generator = Generator(gen_params=gen_params,
                          optimizer=Adam(gen_lr))

    # Discriminator
    discriminator = Discriminator(discr_params=discr_params,
                                  optimizer=Adam(discr_lr))

    # CGAN
    cgan = CGAN(generator=generator,
                discriminator=discriminator)

    # Check generator and discriminator models output
    noise = normalize_noise(tf.random.normal([1, noise_dim]))
    generated_seq = generator.model([tf.convert_to_tensor(train_x.iloc[0].reshape(1, 14, 20)), noise], training=False)
    _logger.info(f'Generator test output: {generated_seq}')
    discr_output = discriminator.model([tf.convert_to_tensor(train_x.iloc[0].reshape(1, 14, 20)), generated_seq]).numpy()
    _logger.info(f'Discriminator test output: {discr_output}')

    # Train CGAN
    cgan.train(train_set, epochs)

    # Save loss curve for generator and discriminator
    pd.DataFrame({'g_loss_curve': cgan.g_loss_curve,
                  'd_loss_curve': cgan.d_loss_curve}).to_csv(RESULTS_PATH / 'сgan_loss.csv')
    _logger.info('сgan loss curve successfully saved.')

    # Save trained generator and discriminator models
    gen_model_file = MODEL_PATH / f"generator_full_model_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.h5"
    generator.model.save(gen_model_file)
    _logger.info(f'generator h5 model successfully saved to {gen_model_file} file.')

    discr_model_file = MODEL_PATH / f"discriminator_full_model_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.h5"
    discriminator.model.save(discr_model_file)
    _logger.info(f'discriminator h5 model successfully saved to {discr_model_file} file.')
