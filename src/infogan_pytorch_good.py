import itertools

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from src.abstract_gan import Latent_GAN
from src.utils import device


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


class Generator(nn.Module):
    def __init__(self, latent_dim, n_categorical=0, n_continuous=0):
        """A generator for mapping a latent space to a sample space.
        The sample space for this generator is single-channel, 32x32 images
        with pixel intensity ranging from -1 to +1.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()

        self.initial_dimension = latent_dim + n_categorical + n_continuous

        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        # Project the input
        self.linear1 = nn.Linear(self.initial_dimension, 448*2*2)
        self.relu = nn.ReLU()
        self.bn1d0 = nn.BatchNorm1d(448*2*2)

        # Convolutions
        self.conv1 = nn.ConvTranspose2d(
                in_channels=448,
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

        self.conv3 = nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1)

        self.conv4 = nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1)

        self.tanh = nn.Tanh()

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""

        intermediate = self.linear1(input_tensor)
        intermediate = self.relu(intermediate)
        intermediate = self.bn1d0(intermediate)

        # reshape
        intermediate = intermediate.view((input_tensor.shape[0], 448, 2, 2))

        intermediate = self.conv1(intermediate)
        intermediate = self.relu(intermediate)
        intermediate = self.bn2d1(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv3(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv4(intermediate)

        output_tensor = self.tanh(intermediate)

        return output_tensor


class Discriminator(nn.Module):
    def __init__(self, n_categorical, n_continuous):
        """A discriminator for discerning real from generated images.
        Images must be single-channel and 28x28 pixels.
        Output activation is Sigmoid.
        """
        super(Discriminator, self).__init__()

        self.n_categorical = n_categorical
        self.n_continuous = n_continuous

        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""

        # BLOCK SHARED BY DISCRIMINATOR AND AUXILIARY
        self.shared_block = nn.Sequential(
            nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # DISCRIMINATOR BLOCK
        self.discriminator_block = nn.Sequential(
            nn.Linear(256*4*4, 1),
            nn.Sigmoid()
        )

        # AUXILIARY BLOCK
        self.auxiliary_block = nn.Sequential(
            nn.Linear(256*4*4, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.auxiliary_block_cat = nn.Sequential(
            nn.Linear(128, self.n_categorical),
            nn.Softmax(dim=1)
        )
        self.auxiliary_block_mu = nn.Linear(128, self.n_continuous)
        self.auxiliary_block_sigma = nn.Linear(128, self.n_continuous)

    def forward(self, input_tensor, as_auxiliary=False):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.shared_block(input_tensor)

        intermediate = intermediate.view((input_tensor.shape[0], 256*4*4))

        if as_auxiliary is False:
            discriminator_output = self.discriminator_block(intermediate)

            return discriminator_output
        else:
            aux_intermediate = self.auxiliary_block(intermediate)

            latent_cat = self.auxiliary_block_cat(aux_intermediate).squeeze()
            latent_continuous_mu = self.auxiliary_block_mu(aux_intermediate).squeeze()
            latent_continuous_sigma = self.auxiliary_block_sigma(aux_intermediate).squeeze().exp()

            return latent_cat, latent_continuous_mu, latent_continuous_sigma


class InfoGAN(Latent_GAN):

    def __init__(self, latent_dim, n_categorical, continuous_list,
                 noise_fn = None, lr_d=0.0004, lr_g=0.0004):
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

        generator = Generator(latent_dim, n_categorical, len(continuous_list)).to(device)
        discriminator = Discriminator(n_categorical, len(continuous_list)).to(device)

        super().__init__(latent_dim,
                         generator=generator,
                         discriminator=discriminator,
                         optim_g=optim.Adam(itertools.chain(*[generator.parameters(),
                                                              discriminator.auxiliary_block.parameters(),
                                                              discriminator.auxiliary_block_cat.parameters(),
                                                              discriminator.auxiliary_block_mu.parameters(),
                                                              discriminator.auxiliary_block_sigma.parameters()]),
                                                              lr=lr_g, betas=(0.5, 0.999)),
                         optim_d=optim.Adam(itertools.chain(*[discriminator.shared_block.parameters(),
                                                              discriminator.discriminator_block.parameters()]),
                                                              lr=lr_d, betas=(0.5, 0.999)),
                         noise_fn=noise_fn)

        self.n_categorical = n_categorical
        self.continuous_list = continuous_list

        self.binary_loss = nn.BCELoss()
        self.categorical_loss = nn.CrossEntropyLoss()
        self.continuous_loss = NormalNLLLoss()

        self.real_labels = None
        self.fake_labels = None

    def create_gen_input(self, current_batch_size, batch_categorical_label=None, batch_c_list=None):

        batch_latent_vec = self.noise_fn(current_batch_size)

        if batch_categorical_label is None:
            # Create categorical latent code
            batch_categorical_label = torch.randint(low=0, high=self.n_categorical, size=(current_batch_size,))  # range [0, n_categorical)
            batch_categorical_label = F.one_hot(batch_categorical_label, num_classes=self.n_categorical).to(device).view(-1, self.n_categorical)

        if batch_c_list is None:
            # Create list of continuous latent code
            batch_c_list = []
            for c_tuple in self.continuous_list:
                low_interval = c_tuple[0]
                high_interval = c_tuple[1]

                c = (high_interval - low_interval) * torch.rand(current_batch_size, 1) + low_interval  # range [low_interval, high_interval)
                batch_c_list.append(c)

        batch_c_tensor = torch.cat(batch_c_list, dim=1).to(device)

        return batch_latent_vec, batch_categorical_label, batch_c_tensor

    def generate_samples(self, num=None, **args):

        if num is not None:
            random_input_gen = self.create_gen_input(num)

        try:
            batch_latent_vec = args["latent_vec"]
        except KeyError:
            if num is None:
                raise ValueError("Must provide either a number of samples or \
                                 a latent_vec, batch_categorical_label and batch_c_tensor in method generate_samples")
            batch_latent_vec = random_input_gen[0]
        
        try:
            batch_categorical_label = args["batch_categorical_label"]
        except KeyError:
            if num is None:
                raise ValueError("Must provide either a number of samples or \
                                 a latent_vec, batch_categorical_label and batch_c_tensor in method generate_samples")
            batch_categorical_label = random_input_gen[1]
        
        try:
            batch_c_tensor = args["batch_c_tensor"]
        except KeyError:
            if num is None:
                raise ValueError("Must provide either a number of samples or \
                                 a latent_vec, batch_categorical_label and batch_c_tensor in method generate_samples")
            batch_c_tensor = random_input_gen[2]

        gen_input = torch.cat([batch_latent_vec, batch_categorical_label, batch_c_tensor], dim=1)
        with torch.no_grad():
            samples = self.generator(gen_input)
        samples = samples.detach().cpu()  # move images to cpu
        return samples

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        self.optim_d.zero_grad()

        pred_real = self.discriminator(real_samples)
        loss_real = self.binary_loss(pred_real, self.real_labels)

        loss_real.backward()

        # generated samples
        batch_latent_vec, batch_categorical_label, batch_c_tensor = self.create_gen_input(current_batch_size)
        gen_input = torch.cat([batch_latent_vec, batch_categorical_label, batch_c_tensor], dim=1)
        with torch.no_grad():
            fake_samples = self.generator(gen_input)

        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.binary_loss(pred_fake, self.fake_labels)

        loss_fake.backward()

        # combine two losses
        loss = (loss_real + loss_fake)
        # loss.backward()
        self.optim_d.step()

        return loss.item()

    def train_step_generator(self, current_batch_size):
        """Train the generator one step and return the loss."""

        self.optim_g.zero_grad()

        # generated samples
        batch_latent_vec, batch_categorical_label, batch_c_tensor = self.create_gen_input(current_batch_size)
        gen_input = torch.cat([batch_latent_vec, batch_categorical_label, batch_c_tensor], dim=1)

        generated = self.generator(gen_input)
        classifications = self.discriminator(generated)

        loss = self.binary_loss(classifications, self.real_labels)
        loss.backward()
        self.optim_g.step()

        return loss.item()

    def train_step_gen_auxiliary(self, current_batch_size):
        """Train the generator one step and return the loss."""

        self.optim_g.zero_grad()

        # GEN LOSS

        batch_latent_vec, batch_categorical_label, batch_c_tensor = self.create_gen_input(current_batch_size)
        gen_input = torch.cat([batch_latent_vec, batch_categorical_label, batch_c_tensor], dim=1)

        generated = self.generator(gen_input)

        classifications = self.discriminator(generated)

        gen_img_loss = self.binary_loss(classifications, self.real_labels)

        # INFO LOSS

        categorical_pred, mu_continuous, sigma_continuous = self.discriminator(generated, as_auxiliary=True)

        # Categorical loss
        cat_loss = self.categorical_loss(categorical_pred.float(), batch_categorical_label.float())

        # # Use Gaussian distributions to represent the output
        # dist = torch.distributions.Normal(loc=mu_continuous, scale=sigma_continuous)
        #
        # # Losses (negative log probability density function as we want to maximize the probability density function)
        # c_losses = torch.mean(-dist.log_prob(batch_c_tensor))

        # Calculate loss for continuous latent code
        c_losses = self.continuous_loss(batch_c_tensor, mu_continuous, sigma_continuous).mean(0)

        # INFOGAN TOTAL LOSS
        # Auxiliary function loss
        q_loss = (1 * cat_loss + 0.1 * c_losses)

        infogan_loss = gen_img_loss + q_loss

        infogan_loss.backward()
        self.optim_g.step()

        return gen_img_loss, q_loss

    def train_step(self, real_data):
        current_batch_size = real_data.size(0)

        # We build labels here so that if the last batch has less samples
        # we don't have to drop it but we can still use it
        # we perform smooth labels
        self.real_labels = torch.ones((current_batch_size, 1), device=device)
        self.real_labels += 0.05 * torch.rand(self.real_labels.size(), device=device)

        self.fake_labels = torch.zeros((current_batch_size, 1), device=device)
        self.fake_labels += 0.05 * torch.rand(self.fake_labels.size(), device=device)

        loss_d = self.train_step_discriminator(real_data, current_batch_size)

        self.real_labels = torch.ones((current_batch_size, 1), device=device)

        self.fake_labels = torch.zeros((current_batch_size, 1), device=device)

        #
        # loss_g = self.train_step_generator(current_batch_size*2)
        #
        # loss_info = self.train_step_infogan(current_batch_size*2, loss_g)

        loss_g, loss_info = self.train_step_gen_auxiliary(current_batch_size)

        loss_g = loss_g.item()
        loss_aux = loss_info.item()

        return {"G_loss": loss_g, "Aux_loss": loss_aux, "D_loss": loss_d}


if __name__ == '__main__':
    """
    GAN architecture inspired by

    https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a

    result with 100 epoch:
    Epoch 100; Elapsed time = 1100s
    G_loss -> 1.9053003065129543, D_loss_real -> 0.23552483283577763, D_loss_fake -> 0.3951658665182743
    """
    latent_dim = 128
    gan = InfoGAN(latent_dim, n_categorical=13, continuous_list=[(-1, 1)])
    gan.train(64, 32, 10, True, True)

    