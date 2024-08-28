import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, BatchNormalization


class Generator:
    _cross_entropy = BinaryCrossentropy(from_logits=False)

    def __init__(self, gen_params, optimizer):
        self.nsample = gen_params['nsample']
        self.n_features = gen_params['n_features']
        self.noise_dim = gen_params['noise_dim']
        self.lstm_units_1 = gen_params['lstm_units_1']
        self.lstm_units_2 = gen_params['lstm_units_2']
        self.dense_units = gen_params['dense_units']
        self.output_size = gen_params['output_size']

        self.optimizer = optimizer

        self.model = self._build_generator()

    def loss_function(self, fake_output):
        return self._cross_entropy(tf.ones_like(fake_output), fake_output)

    def _build_generator(self):
        # Inputs
        context = Input(shape=(self.nsample, self.n_features),
                        name='Context')
        noise = Input(shape=(self.noise_dim,), name='Noise')

        lstm1 = LSTM(units=self.lstm_units_1,
                     input_shape=(self.nsample, self.n_features),
                     name='LSTM1',
                     dropout=.15,
                     recurrent_dropout=.1,
                     return_sequences=True)(context)

        lstm_model = LSTM(units=self.lstm_units_2,
                          input_shape=(self.nsample, self.n_features),
                          dropout=.15,
                          recurrent_dropout=.1,
                          name='LSTM2')(lstm1)

        # Concat model components
        merged = concatenate([lstm_model,
                              noise], name='Merged')

        bn = BatchNormalization(name='bn')(merged)

        hidden_dense = Dense(self.dense_units, activation='relu', name='FC1')(bn)
        # relu = ReLU(name='RELU_G')(hidden_dense)
        # relu = LeakyReLU(alpha=0.2, name='LeakyReLU_G')(hidden_dense)

        output_1 = Dense(self.output_size, activation='sigmoid', name='new_cases_forecast')(hidden_dense)

        gen = Model(inputs=[context, noise], outputs=output_1, name='Generator')

        return gen
