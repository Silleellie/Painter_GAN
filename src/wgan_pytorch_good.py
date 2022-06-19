import torch
import torch.nn as nn
import torch.utils.data
from torch import optim

from src.abstract_gan import LatentGAN
from src.utils import device

"""
Structure of the code mirrors the implementation done for dcgan

Code for both generator and critic is an adaptation of the code from this notebook:
https://www.kaggle.com/code/errorinever/wgan-gp
"""

class Generator(nn.Module):
    """
    https://www.kaggle.com/code/errorinever/wgan-gp
    """
    def __init__(self, channels_noise, channels_img, features_gen):
        """
        :param channels_noise: ``int``, input latent space dimension
        :param channels_img: ``int``,  3 for RGB image or 1 for GrayScale
        :param features_gen: ``int``, num features of generator
        """
        super().__init__()
        self.body = nn.Sequential(
            Generator._default_block(channels_noise, features_gen * 4, 4, 1, 0), 
            Generator._default_block(features_gen * 4, features_gen * 2, 4, 2, 1), 
            Generator._default_block(features_gen * 2, features_gen, 4, 2, 1), 
            nn.ConvTranspose2d(
                features_gen, channels_img, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    @staticmethod
    def _default_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.body(x)


class Critic(nn.Module):
    """
    https://www.kaggle.com/code/errorinever/wgan-gp
    """
    def __init__(self, in_channels, features_d):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Critic._default_block(features_d, features_d * 2, 4, 2, 1),
            Critic._default_block(features_d * 2, features_d * 4, 4, 2, 1),
            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    @staticmethod
    def _default_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.body(x)



class WGAN_GP(LatentGAN):
    def __init__(self, latent_dim=100, init_features: int = 64,
                lr_generator=0.0002, lr_discriminator=0.0002, noise_fn=None, 
                n_repetitions=5, l=10):
        """
        WGAN_GP class

        n_repetitions controls the number of times the critic will be trained for each step (5-10 usually)

        l is the lambda parameter of the gradient penalty (penalty coefficient)
        (default value is 10 as proposed in the improved wgan training paper)
        """
        generator = Generator(latent_dim, 3, init_features).to(device)
        discriminator = Critic(3, init_features).to(device)

        super().__init__(latent_dim, 
                        generator=generator, 
                        discriminator=discriminator, 
                        optim_g=optim.Adam(generator.parameters(), lr=lr_generator, betas=(0.5, 0.999)),
                        optim_d=optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999)),
                        noise_fn=noise_fn)

        self.n_repetitions = n_repetitions
        self.l = l
    
    def compute_gp(self, real_images, fake_images):
        """
        https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead
        """
        batch_size = real_images.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)

        interpolation =  epsilon * real_images + (1 - epsilon) * fake_images

        interp_logits = self.discriminator(interpolation)
        grad_outputs = torch.ones_like(interp_logits)

        gradients = torch.autograd.grad(
            inputs=interpolation,
            outputs=interp_logits,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(len(gradients), -1)
        grad_norm = gradients.norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()
    
    def wasserstein_gp_loss(self, real_images, fake_images, pred_real, pred_fake):
        gp = self.compute_gp(real_images, fake_images)
        # wasserstein loss formula combined with gradient penalty
        return pred_fake.mean() - pred_real.mean() + gp * self.l

    def train_step_generator(self, current_batch_size):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()

        # create noise 
        latent_vec = self.noise_fn(current_batch_size)
        # create fake images based on that noise
        generated = self.generator(latent_vec)
        # classify the created images
        classifications = self.discriminator(generated).reshape(-1)
        loss = -1. * classifications.mean()
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_critic(self, real_samples, current_batch_size):
        """
        Trains the critic n times for each training step (where n = number of repetitions)
        The n number can be modified by changing the n_repetitions attribute
        """
        mean_iter_loss = 0
        for _ in range(self.n_repetitions):
            self.discriminator.zero_grad()

            # classify (networks predicts if images are real or fake) real images using the critic
            pred_real = self.discriminator(real_samples).reshape(-1)

            # generated fake images
            latent_vec = self.noise_fn(current_batch_size)
            with torch.no_grad():
                fake_samples = self.generator(latent_vec)

            # predict the fake images using the critic
            pred_fake = self.discriminator(fake_samples.detach()).reshape(-1)

            loss = self.wasserstein_gp_loss(real_samples, fake_samples, pred_real, pred_fake)
            mean_iter_loss += loss.item()

            loss.backward(retain_graph=True)
            self.optim_d.step()
        return mean_iter_loss / self.n_repetitions

    def train_step(self, real_data):
        current_batch_size = real_data.size(0)
        loss_c = self.train_step_critic(real_data, current_batch_size)
        loss_g = self.train_step_generator(current_batch_size)

        return {"G_loss": loss_g, "C_loss": loss_c}


if __name__ == '__main__':
    gan = WGAN_GP()
    gan.train(32, 32, 20, True, True)

