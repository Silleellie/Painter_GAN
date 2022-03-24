from abc import abstractmethod
from tensorflow import keras as nn
import tensorflow as tf
from typing import List
import numpy as np

class Network:

    def __init__(self, model_name: str = None, layers: List[nn.layers.Layer] = None):
        self.__model = nn.Sequential(layers=[] if layers is None else layers, name=model_name)
        self.__optimizer = None
        self.__loss_metric = None
    
    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, model):
        self.__model = model
    
    @property
    def optimizer(self):
        return self.__optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer
    
    @property
    def loss_metric(self):
        return self.__loss_metric
    
    @loss_metric.setter
    def loss_metric(self, loss_metric):
        self.__loss_metric = loss_metric
    
    def compile(self, optimizer = None, loss_metric = None):
        self.__optimizer = optimizer
        self.__loss_metric = loss_metric
    
    def train(self, tape, loss):
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.__optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.__loss_metric.update_state(loss)

class Generator(Network):

    def __init__(self, layers):
        super().__init__("generator_model", layers)

class Discriminator(Network):

    def __init__(self, layers):
        super().__init__("discriminator_model", layers)

class GAN(nn.Model):

    def __init__(self, generator: Generator, discriminator: Discriminator):
        super(GAN, self).__init__()
        self.__generator = generator
        self.__discriminator = discriminator
    
    def compile(self, loss_fn, g_optimizer = None, g_loss_metric = None, d_optimizer = None, d_loss_metric = None):
        super(GAN, self).compile()
        self.loss_fn = loss_fn
        self.__generator.optimizer = g_optimizer
        self.__discriminator.optimizer = d_optimizer
        self.__generator.loss_metric = g_loss_metric
        self.__discriminator.loss_metric = d_loss_metric
        self.__latent_dim = None
    
    @property
    def latent_dim(self):
        return self.__latent_dim
    
    @property
    def generator(self):
        return self.__generator
    
    @property
    def discriminator(self):
        return self.__discriminator
    
    @abstractmethod
    def process_data(self, real_images):
        raise NotImplementedError
    
    def fit_with_checks(self, dataset, latent_dim, epochs, **kwargs):
        if self.__generator.optimizer is None:
            raise Exception("Define an optimizer function for the generator")
        
        if self.__discriminator.optimizer is None:
            raise Exception("Define an optimizer function for the discriminator")
        
        if self.__generator.loss_metric is None:
            raise Exception("Define a loss metric for the generator")
        
        if self.__discriminator.loss_metric is None:
            raise Exception("Define a loss metric for the discriminator")
        
        self.__latent_dim = latent_dim
        self.fit(dataset, epochs=epochs, **kwargs)
    
    def train_step(self, data: tf.Tensor):
        # combined_images: array containing both the fake and the real images
        # combined_labels: array containing the corresponding label for each image in combined_images (0=real, 1=fake)
        batch_size, combined_images, combined_labels = self.process_data(data)
        
        # discriminator training
        with tf.GradientTape() as tape:
            loss = self.loss_fn(self.__discriminator.model(combined_images), combined_labels)
            self.__discriminator.train(tape, loss)

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.__latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))

        # generator training
        with tf.GradientTape() as tape:
            loss = self.loss_fn(self.__discriminator.model(self.__generator.model(random_latent_vectors)), misleading_labels)
            self.__generator.train(tape, loss)

        return {
            "d_loss": self.__discriminator.loss_metric.result(),
            "g_loss": self.__generator.loss_metric.result(),
        }

class DCGAN(GAN):

    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__(generator, discriminator)
    
    def process_data(self, real_images):
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

    gan = DCGAN(generator=gen, discriminator=disc)

    gan.compile(
        loss_fn=nn.losses.BinaryCrossentropy(),
        g_optimizer=nn.optimizers.Adam(learning_rate=0.0001),
        d_optimizer=nn.optimizers.Adam(learning_rate=0.0001),
        g_loss_metric=nn.metrics.Mean(name="generator_loss"),
        d_loss_metric=nn.metrics.Mean(name="discriminator_loss")
    )

    gan.fit_with_checks(dataset, latent_dim, epochs=20)