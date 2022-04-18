import os
import shutil
from keras.layers import Dense, Input, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, Activation, Cropping2D, \
    ZeroPadding2D, Dropout, Flatten, Reshape

from gan_abstract import PainterGAN
from tensorflow import keras
import tensorflow as tf
import numpy as np


class PainterWGAN(PainterGAN):
    """

    https://keras.io/examples/generative/wgan_gp/

    """
    def __init__(self, discriminator_extra_steps=3, gp_weight=10.0):

        super(PainterWGAN, self).__init__()

        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.latent_dim = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.d_loss_fn = None
        self.g_loss_fn = None

    def build_generator(self, latent_dim, optimizer, loss_fn):

        noise = Input(shape=(latent_dim,))

        x = Dense(4 * 4 * 256, use_bias=False)(noise)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Reshape((4, 4, 256))(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(1, (3, 3), strides=(1, 1), padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Activation("tanh")(x)

        x = Cropping2D((2, 2))(x)

        self.g = keras.models.Model(noise, x, name='generator')

        self.latent_dim = latent_dim
        self.g_optimizer = optimizer
        self.g_loss_fn = loss_fn

    def build_discriminator(self, input_shape, optimizer, loss_fn):

        img_input = Input(shape=input_shape)

        # Zero pad the input to make the input images size to (32, 32, 1).
        x = ZeroPadding2D((2, 2))(img_input)

        x = Conv2D(64, (5, 5), strides=(2, 2), padding="same", use_bias=True)(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(128, (5, 5), strides=(2, 2), padding="same", use_bias=True)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(256, (5, 5), strides=(2, 2), padding="same", use_bias=True)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(512, (5, 5), strides=(2, 2), padding="same", use_bias=True)(x)
        x = LeakyReLU(0.2)(x)

        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(1)(x)

        self.d = keras.models.Model(img_input, x, name="discriminator")

        self.d_optimizer = optimizer
        self.d_loss_fn = loss_fn

    def build_monitor(self, n_img=None):
        class WGANMonitor(keras.callbacks.Callback):
            def __init__(self, num_img, latent_dim):
                self.num_img = num_img
                self.latent_dim = latent_dim

            def on_epoch_end(self, epoch, logs=None):
                random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
                generated_images = self.model.g(random_latent_vectors)
                generated_images.numpy()
                for i in range(self.num_img):
                    converted_image = (generated_images[i] * 127.5) + 127.5
                    img = keras.preprocessing.image.array_to_img(converted_image)
                    img.save("wgan_test/generated_img_%03d_%d.png" % (epoch, i))

        if n_img is None:
            n_img = 3

        self.monitor = WGANMonitor(n_img, self.latent_dim)

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.d(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_images):

        batch_s = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_s, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.g(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.d(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.d(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.d.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer

            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.d.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.g(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.d(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.g.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.g.trainable_variables)
        )
        return {'d_loss': d_loss, 'g_loss': g_loss}


if __name__ == "__main__":

    shutil.rmtree("wgan_test", ignore_errors=True)
    os.makedirs("wgan_test")

    latent_dim = 100
    img_shape = (28, 28, 1)
    batch_size = 128
    epochs = 25

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Add a channel dimension to the images (from 28x28 to 28x28x1)
    # (Since they are gray scale images we add the channel)
    x_train = x_train.astype(np.float32)
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    # reshape to [-1, 1] range
    x_train = (x_train - 127.5) / 127.5

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )

    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

    # Instantiate the WGAN model.
    wgan = PainterWGAN()

    wgan.build_generator(latent_dim, generator_optimizer, generator_loss)
    wgan.build_discriminator(img_shape, discriminator_optimizer, discriminator_loss)
    wgan.build_monitor(n_img=3)

    # Start training the model.
    wgan.fit(x=dataset, batch_size=batch_size, epochs=epochs, callbacks=[wgan.monitor])
