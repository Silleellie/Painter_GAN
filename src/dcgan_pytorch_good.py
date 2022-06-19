import torch
import torch.nn as nn
from torch import optim
import torch.utils.data
import numpy as np

from src.abstract_gan import LatentGAN
from src.utils import device

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        """A generator for mapping a latent space to a sample space.
        The sample space for this generator is single-channel, 32x32 images
        with pixel intensity ranging from -1 to +1.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        # Project the input
        self.linear1 = nn.Linear(self.latent_dim, 512 * 2 * 2)
        self.bn1d1 = nn.BatchNorm1d(512 * 2 * 2)
        self.relu = nn.ReLU()

        # Convolutions
        self.conv1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1)
        self.bn2d1 = nn.BatchNorm2d(256)

        # Convolutions
        self.conv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1)
        self.bn2d2 = nn.BatchNorm2d(128)

        # Convolutions
        self.conv3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1)
        self.bn2d3 = nn.BatchNorm2d(64)

        # Convolutions
        self.conv4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1)
        self.bn2d4 = nn.BatchNorm2d(32)

        self.conv5 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1)
        self.tanh = nn.Tanh()

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = self.linear1(input_tensor)
        intermediate = self.bn1d1(intermediate)
        intermediate = self.relu(intermediate)

        # reshape
        intermediate = intermediate.view((input_tensor.shape[0], 512, 2, 2))

        intermediate = self.conv1(intermediate)
        intermediate = self.bn2d1(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.bn2d2(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv3(intermediate)
        intermediate = self.bn2d3(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv4(intermediate)
        intermediate = self.bn2d4(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv5(intermediate)
        output_tensor = self.tanh(intermediate)
        return output_tensor


class Discriminator(nn.Module):
    def __init__(self):
        """A discriminator for discerning real from generated images.
        Images must be single-channel and 28x28 pixels.
        Output activation is Sigmoid.
        """
        super(Discriminator, self).__init__()
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_2d = nn.Dropout2d(0.25)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True)
        self.bn2d2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(input_tensor)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)
        intermediate = self.bn2d2(intermediate)

        intermediate = self.conv3(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        # reshape
        intermediate = intermediate.view((input_tensor.shape[0], 128 * 8 * 8))
        intermediate = self.linear1(intermediate)
        output_tensor = self.sigmoid(intermediate)

        return output_tensor


class DCGAN(LatentGAN):
    def __init__(self, latent_dim=100,
                 noise_fn=None,lr_d=0.0002, lr_g=0.0002):
        """A very basic DCGAN class for generating MNIST digits
        Args:
            latent_dim: dimension of the latent space
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """

        def noise(x): return torch.normal(mean=0, std=1, size=(x, latent_dim), device=device)

        noise_fn = noise if noise_fn is None else noise_fn

        generator = Generator(latent_dim).to(device)
        discriminator = Discriminator().to(device)

        super().__init__(latent_dim, 
                        generator=generator, 
                        discriminator=discriminator, 
                        optim_g=optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999)),
                        optim_d=optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)),
                        noise_fn=noise_fn)

        self.criterion = nn.BCELoss()
        self.real_labels = None
        self.fake_labels = None

    def train_step_generator(self, current_batch_size):
        """Train the generator one step and return the loss."""
        self.optim_g.zero_grad()

        latent_vec = self.noise_fn(current_batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(classifications, self.real_labels)
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        self.optim_d.zero_grad()

        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, self.real_labels)

        # generated samples
        latent_vec = self.noise_fn(current_batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.fake_labels)

        # combine two losses
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()

    def train_step(self, real_data):
        current_batch_size = real_data.size(0)

        # We build labels here so that if the last batch has less samples
        # we don't have to drop it but we can still use it
        # we perform smooth labels
        self.real_labels = torch.ones((current_batch_size, 1), device=device)
        self.real_labels += 0.05 * torch.rand(self.real_labels.size(), device=device)

        self.fake_labels = torch.zeros((current_batch_size, 1), device=device)
        self.fake_labels += 0.05 * torch.rand(self.fake_labels.size(), device=device)

        loss_g = self.train_step_generator(current_batch_size)

        half_batch_size = int(current_batch_size / 2)

        random_idx = np.random.randint(0, current_batch_size, half_batch_size)
        random_half_batch = real_data[random_idx]

        self.real_labels = torch.ones((half_batch_size, 1), device=device)
        self.real_labels += 0.05 * torch.rand(self.real_labels.size(), device=device)

        self.fake_labels = torch.zeros((half_batch_size, 1), device=device)
        self.fake_labels += 0.05 * torch.rand(self.fake_labels.size(), device=device)

        ldr_, ldf_ = self.train_step_discriminator(random_half_batch, half_batch_size)
        loss_d_real = ldr_
        loss_d_fake = ldf_

        return {"G_loss": loss_g, "D_loss_real": loss_d_real, "D_loss_fake": loss_d_fake}


if __name__ == '__main__':
    """
    GAN architecture inspired by
    
    https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a
    
    result with 100 epoch:
    Epoch 100; Elapsed time = 1100s
    G_loss -> 1.9053003065129543, D_loss_real -> 0.23552483283577763, D_loss_fake -> 0.3951658665182743
    """

    gan = DCGAN()
    gan.train(64, 64, 10, True, True)