import os
import shutil

from keras import Sequential

from src.gan_abstract import PainterGAN
from tensorflow import keras as nn
import tensorflow as tf
import numpy as np


class PainterDCGAN(PainterGAN):

    def __init__(self, loss_fn):
        super().__init__()

        self.loss_fn = loss_fn
        self.latent_dim = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.d_loss_metric = None
        self.g_loss_metric = None

    def build_generator(self, latent_dim, optimizer, loss_fn):
        # weight initializer for Generator following DCGAN paper
        weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        # kernel size==4 and strides==2 because 4 % 2 = 0 so we avoid checkerboard issues
        self.g = Sequential([

            nn.layers.Input(shape=(latent_dim,)),
            nn.layers.Dense(2 * 2 * 512),
            nn.layers.ReLU(),

            # from 1d to 3d
            nn.layers.Reshape((2, 2, 512)),

            # strides 2 so upsample to 8x8 image as output
            nn.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", kernel_initializer=weight_init),
            # nn.layers.BatchNormalization(),
            nn.layers.ReLU(),

            # strides 2 so upsample to 16x16 image as output
            nn.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", kernel_initializer=weight_init),
            nn.layers.BatchNormalization(),
            nn.layers.ReLU(),

            # strides 2 so upsample to 32x32 image as output
            nn.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", kernel_initializer=weight_init),
            nn.layers.BatchNormalization(),
            nn.layers.ReLU(),

            # strides 2 so upsample to 32x32 image as output
            nn.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", kernel_initializer=weight_init),
            # nn.layers.BatchNormalization(),
            nn.layers.ReLU(),

            # 3 as first input since 3 color channels
            nn.layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh"),
        ], name='generator')

        self.latent_dim = latent_dim
        self.g_loss_metric = loss_fn
        self.g_optimizer = optimizer

    def build_discriminator(self, input_shape, optimizer, loss_fn):
        self.d = Sequential([
            nn.layers.Input(shape=(32, 32, 3)),

            nn.layers.Conv2D(32, kernel_size=4, strides=2, padding="same"),
            nn.layers.BatchNormalization(),
            nn.layers.LeakyReLU(alpha=0.2),

            nn.layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            nn.layers.BatchNormalization(),
            nn.layers.LeakyReLU(alpha=0.2),

            nn.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            nn.layers.BatchNormalization(),
            nn.layers.LeakyReLU(alpha=0.2),

            nn.layers.Flatten(),
            nn.layers.Dropout(0.3),
            nn.layers.Dense(1, activation="sigmoid"),
        ], name='discriminator')

        self.d_optimizer = optimizer
        self.d_loss_metric = loss_fn

    def build_monitor(self, n_img=None):
        class DCGANMonitor(nn.callbacks.Callback):
            def __init__(self, num_img, latent_dim):
                self.num_img = num_img
                self.latent_dim = latent_dim

            def on_epoch_end(self, epoch, logs=None):
                random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
                generated_images = self.model.g(random_latent_vectors)
                generated_images.numpy()
                for i in range(self.num_img):
                    converted_image = (generated_images[i] * 127.5) + 127.5
                    img = nn.preprocessing.image.array_to_img(converted_image)
                    img.save("dcgan_test/generated_img_%03d_%d.png" % (epoch, i))

        if n_img is None:
            n_img = 3

        self.monitor = DCGANMonitor(n_img, self.latent_dim)

    @tf.function
    def train_step(self, real_images):

        batch_size = tf.shape(real_images)[0]
        # this are the random vectors, starting point for generation of images
        random_noise_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            # Compute discriminator loss on real images
            pred_real = self.d(real_images, training=True)

            # these are real images so label is 1, but we also perform smooth label
            # it's a simple trick which should improve the process
            real_labels = tf.ones((batch_size, 1))
            # Add random noise to the labels - important trick!
            real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
            d_loss_real = self.loss_fn(real_labels, pred_real)

            # Compute discriminator loss on fake images
            fake_images = self.g(random_noise_vectors)
            pred_fake = self.d(fake_images, training=True)

            # these are fake images (generated by the generator) so label is 0, but we also perform smooth label
            # it's a simple trick which should improve the process
            fake_labels = tf.zeros((batch_size, 1))
            # Add random noise to the labels - important trick!
            fake_labels += 0.05 * tf.random.uniform(tf.shape(fake_labels))
            d_loss_fake = self.loss_fn(fake_labels, pred_fake)

            # total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2

        # Compute discriminator gradients
        grads = tape.gradient(d_loss, self.d.trainable_variables)
        # Update discriminator weights
        self.d_optimizer.apply_gradients(zip(grads, self.d.trainable_variables))

        # We try to fool the discriminator by telling to it that generated images by the generator are real (class 1)
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_images = self.g(random_noise_vectors, training=True)
            pred_fake = self.d(fake_images, training=True)
            g_loss = self.loss_fn(misleading_labels, pred_fake)

        # Compute generator gradients
        grads = tape.gradient(g_loss, self.g.trainable_variables)
        # Update generator weights
        self.g_optimizer.apply_gradients(zip(grads, self.g.trainable_variables))
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


if __name__ == "__main__":
    """
    https://pyimagesearch.com/2021/11/11/get-started-dcgan-for-fashion-mnist/
    
    3 image for each epoch will be saved in dcgan_test folder.
    We try to generate plane images (category 0 of cifar10 dataset)
    
    Generator hard to optimize, I tried a lot of combination in the architecture with no satisfying result
    Maybe different train step?
    """
    shutil.rmtree("dcgan_test", ignore_errors=True)
    os.makedirs("dcgan_test")

    latent_dim = 100
    img_shape = (32, 32, 3)
    batch_size = 8
    epochs = 50

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.reshape((x_train.shape[0], 32, 32, 3)).astype(np.float32)
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 3)).astype(np.float32)

    # we take all images of a plane (category 0) from train set and test set
    plane_images_train = [image for image, category in zip(x_train, y_train) if category == 0]
    plane_images_test = [image for image, category in zip(x_test, y_test) if category == 0]

    train_images = np.concatenate([plane_images_train, plane_images_test])
    dataset = tf.data.Dataset.from_tensor_slices(train_images)

    # each image will be scaled in range [-1, 1] since tanh activation will be used
    dataset = dataset.map(lambda x: x - 127.5 / 127.5)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    dcgan = PainterDCGAN(loss_fn=nn.losses.BinaryCrossentropy(label_smoothing=0.1))

    dcgan.build_generator(latent_dim,
                          optimizer=nn.optimizers.Adam(learning_rate=0.0004, beta_1=0.5),
                          loss_fn=nn.metrics.Mean(name="generator_loss"))

    dcgan.build_discriminator(img_shape,
                              optimizer=nn.optimizers.Adam(learning_rate=0.0003, beta_1=0.5),
                              loss_fn=nn.metrics.Mean(name="discriminator_loss"))

    dcgan.build_monitor(n_img=1)

    dcgan.fit(x=dataset, batch_size=batch_size, epochs=epochs, callbacks=[dcgan.monitor])
