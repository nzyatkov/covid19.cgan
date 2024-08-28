import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, BatchNormalization


class Discriminator:
    _cross_entropy = BinaryCrossentropy(from_logits=False)

    def __init__(self, discr_params, optimizer):
        self.nsample = discr_params['nsample']
        self.n_features = discr_params['n_features']
        self.nsample_forward = discr_params['nsample_forward']
        self.lstm_units_1 = discr_params['lstm_units_1']
        self.lstm_units_2 = discr_params['lstm_units_2']
        self.dense_units = discr_params['dense_units']

        self.optimizer = optimizer

        self.model = self._build_discriminator()

    def loss_function(self, true_output, fake_output):
        true_loss = self._cross_entropy(tf.ones_like(true_output), true_output)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
        return true_loss + fake_loss

    def _build_discriminator(self):
        # Inputs
        context = Input(shape=(self.nsample, self.n_features),
                        name='Context')
        forecast = Input(shape=(self.nsample_forward,), name='Forecast')

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
                              forecast], name='Merged')

        hidden_dense = Dense(self.dense_units, activation='relu', name='FC1')(merged)

        # output_size = 1
        output_1 = Dense(1, activation='sigmoid', name='Output_D')(hidden_dense)

        dis = Model(inputs=[context, forecast], outputs=output_1, name='Discriminator')

        return dis
