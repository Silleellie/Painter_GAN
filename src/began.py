from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from src.abstract_gan import LatentGAN
from src.utils import device


class Decoder(nn.Module):
    def __init__(self, latent_dimension, num_filters):
        super(Decoder, self).__init__()
        self.num_filters = num_filters
        self.h = latent_dimension  # latent dimension is called 'h' in paper

        self._init_modules()

    def _init_modules(self):

        self.h0 = nn.Linear(self.h, self.num_filters * 8 * 8)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.num_filters, 3, 3, 1, 1),
            nn.Tanh()
        )

    def weights_init(self):
        for m in self._modules:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias.data is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        h0 = self.h0(input)

        # reshape
        h0 = h0.view(input.shape[0], self.num_filters, 8, 8)

        # CONV1
        x = self.conv_block1(h0)

        x = torch.cat([x, h0], dim=1)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # CONV2
        x = self.conv_block2(x)

        upsampled_h = F.interpolate(h0, scale_factor=2, mode='nearest')
        x = torch.cat([x, upsampled_h], dim=1)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # CONV3
        x = self.conv_block3(x)

        upsampled_h = F.interpolate(upsampled_h, scale_factor=2, mode='nearest')
        x = torch.cat([x, upsampled_h], dim=1)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv_block4(x)
        x = self.final_conv(x)

        return x


class Encoder(nn.Module):
    def __init__(self, latent_dimension, num_filters):
        super(Encoder, self).__init__()
        self.num_filters = num_filters
        self.h = latent_dimension

        self._init_modules()

    def _init_modules(self):

        # last conv of each layer will take care of subsampling with stride=2

        self.main = nn.Sequential(
            nn.Conv2d(3, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.num_filters, 2 * self.num_filters, 3, 2, 1),
            nn.ELU(inplace=True),

            nn.Conv2d(2 * self.num_filters, 2 * self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(2 * self.num_filters, 3 * self.num_filters, 3, 2, 1),
            nn.ELU(inplace=True),

            nn.Conv2d(3 * self.num_filters, 3 * self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(3 * self.num_filters, 4 * self.num_filters, 3, 2, 1),
            nn.ELU(inplace=True),

            nn.Conv2d(4 * self.num_filters, 4 * self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(4 * self.num_filters, 4 * self.num_filters, 3, 1, 1),
            nn.ELU(inplace=True),

        )

        # output of encoder will be input of encoder which expects input of size 'h'
        self.fc = nn.Linear(4 * 8 * 8 * self.num_filters, self.h)

    def weights_init(self):
        for m in self._modules:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight.data.normal_(0.0, 0.2)
                if m.bias.data is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        x = self.main(input)

        # reshape
        x = x.view(input.shape[0], 4 * self.num_filters * 8 * 8)
        x = self.fc(x)

        return x


# alias for the decoder, since generator has same architecture
class Generator(nn.Module):
    def __init__(self, latent_dimension=100, num_filters=64):
        super(Generator, self).__init__()
        self.dec = Decoder(latent_dimension=latent_dimension, num_filters=num_filters)

    def weights_init(self):
        self.dec.weights_init()

    def forward(self, input):
        out = self.dec(input)

        return out


class Discriminator(nn.Module):
    def __init__(self, latent_dimension=100, num_filters=64):
        super(Discriminator, self).__init__()
        self.enc = Encoder(latent_dimension=latent_dimension, num_filters=num_filters)
        self.dec = Decoder(latent_dimension=latent_dimension, num_filters=num_filters)

    def weights_init(self):
        self.enc.weights_init()
        self.dec.weights_init()

    def forward(self, input):
        out = self.enc(input)
        out = self.dec(out)

        return out


class BEGAN(LatentGAN):
    def __init__(self, latent_dim=64, num_filters=64, noise_fn=None,
                 lr_d=0.0004, lr_g=0.0004, lr_scheduler_class=None):
        """A very basic DCGAN class for generating MNIST digits
        Args:
            generator: a Ganerator network
            discriminator: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """

        def noise(x): return torch.normal(mean=0, std=1, size=(x, latent_dim), device=device)

        noise_fn = noise if noise_fn is None else noise_fn

        lr_scheduler_class = optim.lr_scheduler.ReduceLROnPlateau if lr_scheduler_class is None else lr_scheduler_class
        generator = Generator(latent_dim, num_filters).to(device)

        discriminator = Discriminator(latent_dim, num_filters).to(device)

        optim_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        optim_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))

        super().__init__(latent_dim,
                         generator=generator,
                         discriminator=discriminator,
                         optim_g=optim_g,
                         optim_d=optim_d,
                         noise_fn=noise_fn,
                         lr_scheduler_class=lr_scheduler_class)

        self.criterion = nn.L1Loss()

        self.gamma = 0.5  # controls diversity of generated images
        self.lambda_k = 0.001  # used in paper
        self.Kt = 0.0  # starts at 0

    def train_step_generator(self, fake_samples):
        """Train the generator one step and return the loss."""
        self.optim_g.zero_grad()

        # latent_vec = self.noise_fn(current_batch_size)
        # generated = self.generator(latent_vec)
        classifications = self.discriminator(fake_samples)
        loss = self.criterion(classifications, fake_samples)  # l1_loss

        loss.backward()
        self.optim_g.step()

        return loss

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        self.optim_d.zero_grad()

        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, real_samples)  # l1_loss

        # generated samples
        latent_vec = self.noise_fn(current_batch_size)
        fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples.detach())
        loss_fake = self.criterion(pred_fake, fake_samples.detach())  # l1_loss

        # combine two losses
        loss = loss_real - self.Kt * loss_fake

        loss.backward()
        self.optim_d.step()

        return loss, loss_real, loss_fake, fake_samples

    def train_step(self, real_data):
        current_batch_size = real_data.size(0)

        # ----------
        #  Training discriminator
        # ----------
        loss_d, loss_d_real, loss_d_fake, fake_samples = self.train_step_discriminator(real_data, current_batch_size)

        # ----------
        #  Training generator
        # ----------
        loss_g = self.train_step_generator(fake_samples)

        # ----------------
        # Update weights
        # ----------------
        # Kt update
        balance = (self.gamma * loss_d_real - loss_d_fake).data
        self.Kt = torch.clamp(self.Kt + self.lambda_k * balance, 0, 1).item()

        # Update convergence metric
        M = (loss_d_real + torch.abs(balance)).item()

        # ----------------
        # Update metrics for logging
        # ----------------
        loss_d = loss_d.item()
        loss_g = loss_g.item()

        return {"G_loss": loss_g, "D_loss": loss_d, "Convergence": M, "K": self.Kt}


if __name__ == '__main__':
    """
    BEGAN PAPER
    https://arxiv.org/pdf/1703.10717.pdf
    
    IMPROVED BEGAN PAPER
    https://wlouyang.github.io/Papers/iBegan.pdf
    """
    gan = BEGAN(latent_dim=64, lr_g=0.0001, lr_d=0.0001)
    gan.train_with_default_dataset(batch_size=16,
                                   image_size=64,
                                   epochs=5,
                                   save_imgs_local=True,
                                   wandb_plot=False,
                                   save_model_checkpoints=True,
                                   scheduler_params={'factor': 0.2})
