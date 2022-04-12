from typing import Optional, Callable

from src.gan_abstract import GAN, Generator, Discriminator
from tensorflow import keras as nn
import tensorflow as tf
import numpy as np


class DCGAN(GAN):

    def __init__(self, generator: Generator, discriminator: Discriminator, latent_dim):
        super().__init__(generator, discriminator, latent_dim)
        self.loss_fn: Optional[Callable] = None

    def process_data(self, real_images, labels=None):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        generated_images = self.generator.model(random_latent_vectors)

        # contains both fake and real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # labels to distinguish real from fake images in combined_images (0=real, 1=fake)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        return batch_size, combined_images, labels

    def fit_with_checks(self, dataset, epochs, **kwargs):
        if kwargs.get("loss_fn") is None:
            raise Exception("Define a loss function")

        if kwargs.get("g_optimizer") is None:
            raise Exception("Define an optimizer function for the generator")

        if kwargs.get("d_optimizer") is None:
            raise Exception("Define an optimizer function for the discriminator")

        if kwargs.get("g_loss_metric") is None:
            raise Exception("Define a loss metric for the generator")

        if kwargs.get("d_loss_metric") is None:
            raise Exception("Define a loss metric for the discriminator")

        super(DCGAN, self).compile()
        # pop() instead of get() so that if in future kwargs must be passed to fit() method
        # the dict is cleaned (it's not really a problem, but still)
        self.loss_fn = kwargs.pop("loss_fn", None)
        self.generator.optimizer = kwargs.pop("g_optimizer", None)
        self.discriminator.optimizer = kwargs.pop("d_optimizer", None)
        self.generator.loss_metric = kwargs.pop("g_loss_metric", None)
        self.discriminator.loss_metric = kwargs.pop("d_loss_metric", None)

        self.fit(dataset, epochs=epochs)

    def train_step(self, data: tf.Tensor):
        # combined_images: array containing both the fake and the real images
        # combined_labels: array containing the corresponding label for each image in combined_images (0=real, 1=fake)
        batch_size, combined_images, combined_labels = self.process_data(data)

        # discriminator training
        with tf.GradientTape() as tape:
            loss = self.loss_fn(self.__discriminator.model(combined_images), combined_labels)
            self.__discriminator.train(tape, loss)

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))

        # generator training
        with tf.GradientTape() as tape:
            loss = self.loss_fn(self.discriminator.model(self.generator.model(random_latent_vectors)),
                                misleading_labels)
            self.generator.train(tape, loss)

        return {
            "d_loss": self.discriminator.loss_metric.result(),
            "g_loss": self.generator.loss_metric.result(),
        }

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = nn.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])

    # Scale the pixel values to [0, 1] range
    all_digits = all_digits.astype(np.float32) / 255.0
    # Add a channel dimension to the images (from 28x28 to 28x28x1)
    # (Since they are gray scale images we add the channel)
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))

    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    image_size = 28
    latent_dim = 128

    gen = Generator([
        nn.layers.InputLayer((latent_dim,)),
        nn.layers.Dense(7 * 7 * 128),
        nn.layers.LeakyReLU(alpha=0.2),
        nn.layers.Reshape((7, 7, 128)),
        nn.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        nn.layers.LeakyReLU(alpha=0.2),
        nn.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        nn.layers.LeakyReLU(alpha=0.2),
        nn.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ])

    disc = Discriminator([
        nn.layers.InputLayer((28, 28, 1)),
        nn.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        nn.layers.LeakyReLU(alpha=0.2),
        nn.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        nn.layers.LeakyReLU(alpha=0.2),
        nn.layers.GlobalMaxPooling2D(),
        nn.layers.Dense(1),
    ])

    gan = DCGAN(generator=gen, discriminator=disc, latent_dim=latent_dim)

    gan.fit_with_checks(dataset, epochs=2,
                        loss_fn=nn.losses.BinaryCrossentropy(),
                        g_optimizer=nn.optimizers.Adam(learning_rate=0.0001),
                        d_optimizer=nn.optimizers.Adam(learning_rate=0.0001),
                        g_loss_metric=nn.metrics.Mean(name="generator_loss"),
                        d_loss_metric=nn.metrics.Mean(name="discriminator_loss"))
