import numpy as np
import tensorflow as tf
from train_model.constants_train_model import noise_dim
from logger.logger import get_logger
from utils.common import normalize_noise


class CGAN:
    g_loss_curve = []
    d_loss_curve = []
    g_loss_curve_val = []
    d_loss_curve_val = []

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    @tf.function
    def train_step(self, context, true_forecast):
        # generate the random input for the generator
        noise = normalize_noise(tf.random.normal([context.shape[0], noise_dim]))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # get the generator output
            generated_forecast = self.generator.model([context, noise], training=True)

            # collect discriminator decisions regarding real and fake input
            true_output = self.discriminator.model([context, true_forecast], training=True)
            fake_output = self.discriminator.model([context, generated_forecast], training=True)

            # compute the loss for each model
            gen_loss = self.generator.loss_function(fake_output)  # ---> min
            disc_loss = self.discriminator.loss_function(true_output, fake_output)  # ---> min

        # compute the gradients for each loss with respect to the model variables
        grad_generator = gen_tape.gradient(gen_loss, self.generator.model.trainable_variables)
        grad_discriminator = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)

        # apply the gradient to complete the backpropagation step
        self.generator.optimizer.apply_gradients(zip(grad_generator, self.generator.model.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(grad_discriminator, self.discriminator.model.trainable_variables))

        return gen_loss, disc_loss

    def train(self, train_set, epochs):
        _logger = get_logger()
        _logger.info(f'CGAN training started...')

        for epoch in range(epochs):
            g_loss_ave = []
            d_loss_ave = []
            for train_set_batch in train_set:
                g_loss, d_loss = self.train_step(train_set_batch[0], train_set_batch[1])
                g_loss_ave.append(g_loss)
                d_loss_ave.append(d_loss)

            g_loss_ave = np.mean(g_loss_ave)
            d_loss_ave = np.mean(d_loss_ave)

            self.g_loss_curve.append(g_loss_ave)
            self.d_loss_curve.append(d_loss_ave)

            # val_g_loss_ave, val_d_loss_ave = val_epoch_loss()
            # g_loss_curve_val.append(val_g_loss_ave)
            # d_loss_curve_val.append(val_d_loss_ave)

            if epoch % 1 == 0:
                _logger.info(f'{epoch:6,.0f} | d_loss: {d_loss_ave:6.4f} | g_loss: {g_loss_ave:6.4f}')

            # Save the model every 10 EPOCHS
            # if (epoch + 1) % save_every == 0:
            #    checkpoint.save(file_prefix=checkpoint_prefix)
