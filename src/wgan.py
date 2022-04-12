from typing import Optional, Callable

from gan_abstract import GAN, Generator, Discriminator
from tensorflow import keras as nn
import tensorflow as tf
import numpy as np


class WGAN(GAN):
    """

    https://keras.io/examples/generative/wgan_gp/

    """
    def __init__(self, generator, discriminator, latent_dim, discriminator_extra_steps=3, gp_weight=10.0):

        super(WGAN, self).__init__(generator, discriminator, latent_dim)
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.generator_optimizer: Optional[nn.optimizers] = None
        self.discriminator_optimizer: Optional[nn.optimizers] = None
        self.generator_loss_fn: Optional[Callable] = None
        self.discriminator_loss_fn: Optional[Callable] = None

    def fit_with_checks(self, dataset, epochs, **kwargs):
        if kwargs.get("g_optimizer") is None:
            raise Exception("Define an optimizer function for the generator")

        if kwargs.get("d_optimizer") is None:
            raise Exception("Define an optimizer function for the discriminator")

        if kwargs.get("g_loss_fn") is None:
            raise Exception("Define a loss function for the generator")

        if kwargs.get("d_loss_fn") is None:
            raise Exception("Define a loss function for the discriminator")

        super(WGAN, self).compile()
        # pop() instead of get() so that if in future kwargs must be passed to fit() method
        # the dict is cleaned (it's not really a problem, but still)
        self.generator_optimizer = kwargs.pop("g_optimizer", None)
        self.discriminator_optimizer = kwargs.pop("d_optimizer", None)
        self.generator_loss_fn = kwargs.pop("g_loss_fn", None)
        self.discriminator_loss_fn = kwargs.pop("d_loss_fn", None)

        self.fit(dataset, epochs=epochs)

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
            pred = self.discriminator.model(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def process_data(self, real_images, labels=None):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        return real_images, batch_size

    def train_step(self, data: tf.Tensor):

        real_images, batch_size = self.process_data(data)

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
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator.model(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator.model(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator.model(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.discriminator_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.model.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.discriminator_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.model.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator.model(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator.model(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.generator_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.model.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.model.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


if __name__ == "__main__":
    noise_dim = 128
    img_shape = (28, 28, 1)
    batch_size = 512

    (x_train, y_train), (x_test, y_test) = nn.datasets.mnist.load_data()

    # Scale the pixel values to [0, 1] range
    x_train = x_train.astype(np.float32) / 255.0
    # Add a channel dimension to the images (from 28x28 to 28x28x1)
    # (Since they are gray scale images we add the channel)
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_train = (x_train - 127.5) / 127.5

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    gen = Generator([

        nn.layers.InputLayer((noise_dim,)),

        nn.layers.Dense(4 * 4 * 256, use_bias=False),
        nn.layers.BatchNormalization(),
        nn.layers.LeakyReLU(0.2),
        nn.layers.Reshape((4, 4, 256)),

        nn.layers.UpSampling2D((2, 2)),
        nn.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", use_bias=False),
        nn.layers.BatchNormalization(),
        nn.layers.LeakyReLU(0.2),

        nn.layers.UpSampling2D((2, 2)),
        nn.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", use_bias=False),
        nn.layers.BatchNormalization(),
        nn.layers.LeakyReLU(0.2),

        nn.layers.UpSampling2D((2, 2)),
        nn.layers.Conv2D(1, (3, 3), strides=(1, 1), padding="same", use_bias=False),
        nn.layers.BatchNormalization(),
        nn.layers.Activation("tanh"),

        nn.layers.Cropping2D((2, 2))

    ])

    critic = Discriminator([

        nn.layers.InputLayer(img_shape),
        # Zero pad the input to make the input images size to (32, 32, 1).
        nn.layers.ZeroPadding2D((2, 2)),

        nn.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", use_bias=True),
        nn.layers.LeakyReLU(0.2),

        nn.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same", use_bias=True),
        nn.layers.LeakyReLU(0.2),
        nn.layers.Dropout(0.3),

        nn.layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same", use_bias=True),
        nn.layers.LeakyReLU(0.2),
        nn.layers.Dropout(0.3),

        nn.layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same", use_bias=True),
        nn.layers.LeakyReLU(0.2),

        nn.layers.Flatten(),
        nn.layers.Dropout(0.2),
        nn.layers.Dense(1)
    ])

    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = nn.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = nn.optimizers.Adam(
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

    # Set the number of epochs for trainining.
    epochs = 2

    # Instantiate the WGAN model.
    wgan = WGAN(
        generator=gen,
        discriminator=critic,
        latent_dim=noise_dim,
        discriminator_extra_steps=3
    )

    # Start training the model.
    wgan.fit_with_checks(dataset, epochs=epochs,
                         d_optimizer=discriminator_optimizer,
                         g_optimizer=generator_optimizer,
                         g_loss_fn=generator_loss,
                         d_loss_fn=discriminator_loss,
                         )
