import itertools
import os
import shutil

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from time import time
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset

torch.cuda.empty_cache()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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
        self.linear1 = nn.Linear(self.initial_dimension, 512*2*2)
        self.relu = nn.ReLU()
        self.bn1d0 = nn.BatchNorm1d(512*2*2)

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
        intermediate = intermediate.view((-1, 512, 2, 2))

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
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1)

        self.conv2 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1)
        self.bn2d2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1)
        self.bn2d3 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten()

        # DISCRIMINATOR BLOCK
        self.linear1_disc = nn.Linear(256*4*4, 1)
        self.sigmoid = nn.Sigmoid()

        # AUXILIARY BLOCK
        self.linear1_aux = nn.Linear(256*4*4, 128)
        self.bn2d4_aux = nn.BatchNorm1d(128)
        self.linear_cat_aux = nn.Linear(128, self.n_categorical)
        self.softmax = nn.Softmax()

        # FC layer for continuous output:
        # Gaussian distribution mean (continuous output)
        # Gaussian distribution standard deviation (exponential activation to ensure the value is positive)
        self.linear_continuous_aux = nn.Linear(128, self.n_continuous)
        self.exp_activation = lambda x: torch.exp(x)

    def forward(self, input_tensor, also_auxiliary_output=False):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(input_tensor)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn2d2(intermediate)

        intermediate = self.conv3(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn2d3(intermediate)

        intermediate = self.flatten(intermediate)

        intermediate = intermediate.view((-1, 256*4*4))
        disc_intermediate = self.linear1_disc(intermediate)

        # prediction, discriminator output
        disc_output_tensor = self.sigmoid(disc_intermediate)

        if also_auxiliary_output is True:
            # aux output
            aux_intermediate = self.linear1_aux(intermediate)
            aux_intermediate = self.bn2d4_aux(aux_intermediate)

            aux_discrete_intermediate = self.linear_cat_aux(aux_intermediate)
            aux_discrete_output = self.softmax(aux_discrete_intermediate)

            aux_mu_output = self.linear_continuous_aux(aux_intermediate)

            aux_sigma_intermediate = self.linear_continuous_aux(aux_intermediate)
            aux_sigma_output = self.exp_activation(aux_sigma_intermediate)

            return disc_output_tensor, aux_discrete_output, (aux_mu_output, aux_sigma_output)
        else:
            return disc_output_tensor


class Auxiliary(nn.Module):

    def __init__(self, n_categorical, n_continuous):
        """A discriminator for discerning real from generated images.
        Images must be single-channel and 28x28 pixels.
        Output activation is Sigmoid.
        """
        self.n_categorical = n_categorical
        self.n_continuous = n_continuous

        super(Auxiliary, self).__init__()
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1)

        self.conv2 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1)
        self.bn2d2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1)

        self.bn2d3 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256*4*4, 128)

        self.bn2d4 = nn.BatchNorm1d(128)

        self.linear_cat = nn.Linear(128, self.n_categorical)
        self.softmax = nn.Softmax()

        # FC layer for continuous output:
        # Gaussian distribution mean (continuous output)
        # Gaussian distribution standard deviation (exponential activation to ensure the value is positive)
        self.linear_continuous = nn.Linear(128, self.n_continuous)

        self.exp_activation = lambda x: torch.exp(x)

    def forward(self, input_tensor):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(input_tensor)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn2d2(intermediate)

        intermediate = self.conv3(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn2d3(intermediate)

        intermediate = self.flatten(intermediate)

        intermediate = intermediate.view((-1, 256*4*4))
        intermediate = self.linear1(intermediate)
        intermediate = self.bn2d4(intermediate)

        discrete_intermediate = self.linear_cat(intermediate)
        discrete_output = self.softmax(discrete_intermediate)

        mu_output = self.linear_continuous(intermediate)

        sigma_intermediate = self.linear_continuous(intermediate)
        sigma_output = self.exp_activation(sigma_intermediate)

        return discrete_output, mu_output, sigma_output


class InfoGAN:

    def __init__(self, latent_dim, noise_fn, dataloader, n_categorical, continuous_list,
                 batch_size=32, lr_d=0.0004, lr_g=0.0004, lr_q=0.0004, device: torch.device ='cpu'):
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
        self.n_categorical = n_categorical
        self.continuous_list = continuous_list

        self.generator = Generator(latent_dim, self.n_categorical, len(continuous_list)).to(device)
        self.generator.apply(weights_init)

        self.discriminator = Discriminator(self.n_categorical, len(continuous_list)).to(device)
        self.discriminator.apply(weights_init)

        self.auxiliary = Auxiliary(self.n_categorical, len(continuous_list)).to(device)
        self.auxiliary.apply(weights_init)

        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.device = device

        self.binary_loss = nn.BCELoss()
        self.categorical_loss = nn.CrossEntropyLoss()

        self.optim_d = optim.Adam(self.discriminator.parameters(),
                                  lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.999))
        self.optim_q = optim.Adam(self.auxiliary.parameters(),
                                  lr=lr_q, betas=(0.5, 0.999))

        self.real_labels = None
        self.fake_labels = None

    def create_gen_input(self, current_batch_size, batch_categorical_label=None, batch_c_list=None):

        batch_latent_vec = self.noise_fn(current_batch_size)

        if batch_categorical_label is None:
            # Create categorical latent code
            batch_categorical_label = torch.randint(low=0, high=self.n_categorical, size=(current_batch_size,))  # range [0, n_categorical)
            batch_categorical_label = F.one_hot(batch_categorical_label, num_classes=self.n_categorical).to(self.device).view(-1, self.n_categorical)

        if batch_c_list is None:
            # Create list of continuous latent code
            batch_c_list = []
            for c_tuple in self.continuous_list:
                low_interval = c_tuple[0]
                high_interval = c_tuple[1]

                c = (high_interval - low_interval) * torch.rand(current_batch_size, 1) + low_interval  # range [low_interval, high_interval)
                batch_c_list.append(c)

        batch_c_tensor = torch.cat(batch_c_list, dim=1).to(self.device)

        return batch_latent_vec, batch_categorical_label, batch_c_tensor

    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.
        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.
        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None
        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        batch_latent_vec, batch_categorical_label, batch_c_tensor = self.create_gen_input(num)
        gen_input = torch.cat([batch_latent_vec, batch_categorical_label, batch_c_tensor], dim=1)
        with torch.no_grad():
            samples = self.generator(gen_input)
        samples = samples.detach().cpu()  # move images to cpu
        return samples

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        self.optim_d.zero_grad()

        pred_real = self.discriminator(real_samples, also_auxiliary_output=False)
        loss_real = self.binary_loss(pred_real, self.real_labels)

        # generated samples
        batch_latent_vec, batch_categorical_label, batch_c_tensor = self.create_gen_input(current_batch_size)
        gen_input = torch.cat([batch_latent_vec, batch_categorical_label, batch_c_tensor], dim=1)
        fake_samples = self.generator(gen_input)

        pred_fake = self.discriminator(fake_samples, also_auxiliary_output=False)
        loss_fake = self.binary_loss(pred_fake, self.fake_labels)

        # combine two losses
        loss = (loss_real + loss_fake)
        loss.backward()
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

    def train_step_infogan(self, current_batch_size, gen_loss):

        self.optim_i.zero_grad()

        # train info loss

        batch_latent_vec, batch_categorical_label, batch_c_tensor = self.create_gen_input(current_batch_size)
        gen_input = torch.cat([batch_latent_vec, batch_categorical_label, batch_c_tensor], dim=1)

        generated = self.generator(gen_input)

        classifications, categorical_pred, (mu_continuous, sigma_continuous) = self.discriminator(generated,
                                                                                                  also_auxiliary_output=True)
        # Categorical loss
        cat_loss = self.categorical_loss(categorical_pred.float(), batch_categorical_label.float())

        # Use Gaussian distributions to represent the output
        dist = torch.distributions.Normal(loc=mu_continuous, scale=sigma_continuous)

        # Losses (negative log probability density function as we want to maximize the probability density function)
        c_losses = torch.mean(-dist.log_prob(batch_c_tensor))

        # INFOGAN TOTAL LOSS
        # Auxiliary function loss
        q_loss = (1 * cat_loss + 0.1 * c_losses)

        infogan_loss = gen_loss + q_loss
        infogan_loss.backward()
        self.optim_i.step()

        return infogan_loss

    def train_step_gen_auxiliary(self, current_batch_size):
        """Train the generator one step and return the loss."""


        # GEN LOSS

        batch_latent_vec, batch_categorical_label, batch_c_tensor = self.create_gen_input(current_batch_size)
        gen_input = torch.cat([batch_latent_vec, batch_categorical_label, batch_c_tensor], dim=1)

        generated = self.generator(gen_input)

        classifications = self.discriminator(generated, also_auxiliary_output=False)

        gen_img_loss = self.binary_loss(classifications, self.real_labels)

        # INFO LOSS

        categorical_pred, mu_continuous, sigma_continuous = self.auxiliary(generated)

        # Categorical loss
        cat_loss = self.categorical_loss(categorical_pred.float(), batch_categorical_label.float())

        # Use Gaussian distributions to represent the output
        dist = torch.distributions.Normal(loc=mu_continuous, scale=sigma_continuous)

        # Losses (negative log probability density function as we want to maximize the probability density function)
        c_losses = torch.mean(-dist.log_prob(batch_c_tensor))

        # INFOGAN TOTAL LOSS
        # Auxiliary function loss
        q_loss = (1 * cat_loss + 0.1 * c_losses)

        infogan_loss = gen_img_loss + q_loss

        self.optim_g.zero_grad()
        infogan_loss.backward()

        self.optim_q.zero_grad()
        q_loss = q_loss.detach()
        q_loss.requires_grad = True
        q_loss.backward()

        self.optim_g.step()
        self.optim_q.step()

        return gen_img_loss, q_loss

    def train_epoch(self):
        """Train both networks for one epoch and return the losses.
        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        loss_g_running, loss_d_running, loss_auxiliary_running = 0, 0, 0
        for batch, (real_samples, _) in enumerate(self.dataloader):

            current_batch_size = real_samples.size(0)
            real_samples = real_samples.to(self.device)

            # We build labels here so that if the last batch has less samples
            # we don't have to drop it but we can still use it
            # we perform smooth labels
            self.real_labels = torch.ones((current_batch_size, 1), device=self.device)
            self.real_labels += 0.05 * torch.rand(self.real_labels.size(), device=self.device)

            self.fake_labels = torch.zeros((current_batch_size, 1), device=self.device)
            self.fake_labels += 0.05 * torch.rand(self.fake_labels.size(), device=self.device)

            loss_d_running += self.train_step_discriminator(real_samples, current_batch_size)

            #
            # loss_g = self.train_step_generator(current_batch_size*2)
            #
            # loss_info = self.train_step_infogan(current_batch_size*2, loss_g)

            loss_g, loss_info = self.train_step_gen_auxiliary(current_batch_size)

            loss_g_running += loss_g
            loss_auxiliary_running += loss_info

        n_batches = len(self.dataloader)
        loss_g_running /= n_batches
        loss_d_running /= n_batches
        loss_auxiliary_running /= n_batches

        return loss_g_running, loss_d_running, loss_auxiliary_running


if __name__ == '__main__':
    """
    GAN architecture inspired by

    https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a

    result with 100 epoch:
    Epoch 100; Elapsed time = 1100s
    G_loss -> 1.9053003065129543, D_loss_real -> 0.23552483283577763, D_loss_fake -> 0.3951658665182743
    """
    output_dir = "../output/infogan_test_pytorch"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    image_size = 32
    batch_size = 64
    epochs = 200
    latent_dim = 100

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    train = dset.CIFAR10(root='../dataset/cifar10',
                         transform=transforms.Compose([
                             transforms.Resize(image_size),
                             transforms.CenterCrop(image_size),
                             transforms.ToTensor(),

                             # normalizes images in range [-1,1]
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                         ]),
                         train=True,
                         download=True
                         )

    test = dset.CIFAR10(root='../dataset/cifar10',
                        transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),

                            # normalizes images in range [-1,1]
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]),
                        train=False,
                        download=True
                        )

    dataset = ConcatDataset([train, test])

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    # noise_fn is the function which will sample the latent vector from the gaussian distribution in this case
    noise_fn = lambda x: torch.normal(mean=0, std=1, size=(x, latent_dim), device=device)
    gan = InfoGAN(latent_dim, noise_fn, dataloader, n_categorical=10, continuous_list=[(-1, 1)],
                  batch_size=batch_size, device=device,
                  lr_g=5e-4, lr_d=2e-4, lr_q=2e-4)

    print("Started training...")
    for i in range(epochs):
        g_loss, d_loss, q_loss = gan.train_epoch()

        print(
            "[Epoch %d/%d] [G loss: %f] [D loss: %f] [Q loss: %f]"
            % (i + 1, epochs, g_loss, d_loss, q_loss)
        )

        # save grid of 64 imgs for each epoch, to see generator progress
        images = gan.generate_samples(num=64)
        ims = vutils.make_grid(images, normalize=True)
        plt.axis("off")
        plt.title(f"Epoch {i + 1}")
        plt.imshow(np.transpose(ims, (1, 2, 0)))
        plt.savefig(f'{output_dir}/epoch{i + 1}.png')