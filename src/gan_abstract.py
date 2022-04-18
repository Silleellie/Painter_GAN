from abc import abstractmethod

import keras
from tensorflow import keras as nn
import tensorflow as tf
from typing import List, Optional


class Network:

    def __init__(self, model_name: str = None, layers: List[nn.layers.Layer] = None):
        self.model = nn.Sequential(layers=[] if layers is None else layers, name=model_name)
        self.optimizer = None
        self.loss_metric = None

    def compile(self, optimizer=None, loss_metric=None):
        self.optimizer = optimizer
        self.loss_metric = loss_metric

    def train(self, tape, loss):
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.loss_metric.update_state(loss)


class Generator(Network):

    def __init__(self, layers):
        super().__init__("generator_model", layers)


class Discriminator(Network):

    def __init__(self, layers):
        super().__init__("discriminator_model", layers)


class GAN(nn.Model):

    def __init__(self, generator: Generator, discriminator: Discriminator, latent_dim: int):
        super(GAN, self).__init__()
        self.__generator = generator
        self.__discriminator = discriminator
        self.latent_dim = latent_dim

    @property
    def generator(self):
        return self.__generator

    @property
    def discriminator(self):
        return self.__discriminator

    @abstractmethod
    def process_data(self, real_images, labels=None):
        raise NotImplementedError

    @abstractmethod
    def fit_with_checks(self, dataset, latent_dim, epochs, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, data: tf.Tensor):
        raise NotImplementedError


class PainterGAN(keras.Model):

    def __init__(self):
        super().__init__()

        super().compile()

        # evaluated upon call of build_generator()
        self.g: Optional[nn.Sequential] = None

        # evaluated upon call of build_discriminator()
        self.d: Optional[nn.Sequential] = None

        # evaluated upon call of build_monitor()
        self.monitor: Optional[nn.callbacks.Callback] = None

    @abstractmethod
    def build_generator(self, latent_dim, optimizer, loss_fn):
        raise NotImplementedError

    def build_discriminator(self, input_shape, optimizer, loss_fn):
        raise NotImplementedError

    def build_monitor(self, n_img: int = None):
        raise NotImplementedError
