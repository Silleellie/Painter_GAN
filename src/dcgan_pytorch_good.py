import torch
import torch.nn as nn
from torch import optim
import torch.utils.data

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

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim,
                               1024,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            # Convolutions
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Convolutions
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Convolutions
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Convolutions
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1),

            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)


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
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):

        return self.main(input_tensor)


class DCGAN(LatentGAN):
    def __init__(self, latent_dim=100,
                 noise_fn=None, lr_d=0.0002, lr_g=0.0002):
        """A very basic DCGAN class for generating MNIST digits
        Args:
            latent_dim: dimension of the latent space
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """

        def noise(x): return torch.normal(mean=0, std=1, size=(x, latent_dim, 1, 1), device=device)

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

    def train_step_generator(self, fake_samples, current_batch_size):
        """Train the generator one step and return the loss."""
        self.optim_g.zero_grad()

        # latent_vec = self.noise_fn(current_batch_size)
        # generated = self.generator(latent_vec)
        classifications = self.discriminator(fake_samples).view(-1)
        loss = self.criterion(classifications, self.real_labels)
        loss.backward()
        self.optim_g.step()

        return loss.mean().item()

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        self.optim_d.zero_grad()

        pred_real = self.discriminator(real_samples).view(-1)
        loss_real = self.criterion(pred_real, self.real_labels)

        loss_real.backward()

        # generated samples
        latent_vec = self.noise_fn(current_batch_size)
        fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples.detach()).view(-1)
        loss_fake = self.criterion(pred_fake, self.fake_labels)

        loss_fake.backward()

        # combine two losses. No actual mean since it's just a scalar
        loss = loss_real.mean().item() + loss_fake.mean().item()

        self.optim_d.step()

        return fake_samples, loss

    def train_step(self, real_data):
        current_batch_size = real_data.size(0)

        # We build labels here so that if the last batch has less samples
        # we don't have to drop it but we can still use it
        # we perform smooth labels
        self.real_labels = torch.full((current_batch_size,), 1., dtype=torch.float, device=device)
        self.fake_labels = torch.full((current_batch_size,), 0., dtype=torch.float, device=device)

        fake_samples, loss_d = self.train_step_discriminator(real_data, current_batch_size)

        loss_g = self.train_step_generator(fake_samples, current_batch_size)

        return {"G_loss": loss_g, "D_loss": loss_d}


if __name__ == '__main__':
    """
    GAN architecture inspired by
    
    https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a
    
    result with 100 epoch:
    Epoch 100; Elapsed time = 1100s
    G_loss -> 1.9053003065129543, D_loss_real -> 0.23552483283577763, D_loss_fake -> 0.3951658665182743
    """

    gan = DCGAN(latent_dim=100)
    gan.train_with_default_dataset(batch_size=128,
                                   image_size=64,
                                   epochs=5,
                                   save_model_checkpoints=False,
                                   save_imgs_local=True,
                                   wandb_plot=False)
