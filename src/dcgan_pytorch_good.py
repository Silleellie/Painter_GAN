import os
import shutil

import torch
import torch.nn as nn
from torch import optim
from time import time
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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
        self.linear1 = nn.Linear(self.latent_dim, 256*4*4)
        self.bn1d1 = nn.BatchNorm1d(256*4*4)
        self.relu = nn.ReLU()

        # Convolutions
        self.conv1 = nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1)
        self.bn2d1 = nn.BatchNorm2d(128)

        # Convolutions
        self.conv2 = nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1)
        self.bn2d2 = nn.BatchNorm2d(64)

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
        intermediate = self.bn1d1(intermediate)
        intermediate = self.relu(intermediate)

        # reshape
        intermediate = intermediate.view((-1, 256, 4, 4))

        intermediate = self.conv1(intermediate)
        intermediate = self.bn2d1(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.bn2d2(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv4(intermediate)
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
        self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True)
        self.bn2d1 = nn.BatchNorm2d(64)
        self.leaky_relu = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True)
        self.bn2d2 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.dropout_2d = nn.Dropout2d(0.3)
        self.linear1 = nn.Linear(128*8*8, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(input_tensor)
        intermediate = self.bn2d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.bn2d2(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = intermediate.view((-1, 128*8*8))
        intermediate = self.linear1(intermediate)
        output_tensor = self.sigmoid(intermediate)

        return output_tensor


class DCGAN:
    def __init__(self, latent_dim, noise_fn, dataloader,
                 batch_size=32, lr_d=0.002, lr_g=0.002, device: torch.device ='cpu'):
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
        self.generator = Generator(latent_dim).to(device)
        self.generator.apply(weights_init)

        self.discriminator = Discriminator().to(device)
        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.BCELoss()
        self.optim_d = optim.Adam(self.discriminator.parameters(),
                                  lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.999))
        self.real_labels = None
        self.fake_labels = None

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
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.detach().cpu()  # move images to cpu
        return samples

    def train_step_generator(self, current_batch_size):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()

        latent_vec = self.noise_fn(current_batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(classifications, self.real_labels)
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        self.discriminator.zero_grad()

        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, self.real_labels)

        # generated samples
        latent_vec = self.noise_fn(current_batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.fake_labels)

        # combine
        loss = (loss_real + loss_fake)
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()

    def train_epoch(self):
        """Train both networks for one epoch and return the losses.
        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        for batch, (real_samples, _) in enumerate(self.dataloader):

            current_batch_size = real_samples.size(0)
            # We build labels here so that if the last batch has less samples
            # we don't have to drop it but we can still use it
            # we perform smooth labels
            self.real_labels = torch.ones((current_batch_size, 1), device=device)
            self.real_labels += 0.05 * torch.rand(self.real_labels.size(), device=device)

            self.fake_labels = torch.zeros((current_batch_size, 1), device=device)
            self.fake_labels += 0.05 * torch.rand(self.fake_labels.size(), device=device)

            real_samples = real_samples.to(self.device)
            ldr_, ldf_ = self.train_step_discriminator(real_samples, current_batch_size)
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_
            loss_g_running += self.train_step_generator(current_batch_size)

        n_batches = len(self.dataloader)
        loss_g_running /= n_batches
        loss_d_real_running /= n_batches
        loss_d_fake_running /= n_batches

        return loss_g_running, (loss_d_real_running, loss_d_fake_running)


if __name__ == '__main__':
    """
    GAN architecture inspired by
    
    https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a
    
    """

    shutil.rmtree("dcgan_test_pytorch", ignore_errors=True)
    os.makedirs("dcgan_test_pytorch")

    image_size = 32
    batch_size = 64
    epochs = 100
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

    train_airplane_indexes = [i for i, target in enumerate(train.targets) if target == 0]
    train_airplane = torch.utils.data.Subset(train, train_airplane_indexes)

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
    test_airplane_indexes = [i for i, target in enumerate(test.targets) if target == 0]
    test_airplane = torch.utils.data.Subset(test, test_airplane_indexes)

    dataset = ConcatDataset([train_airplane, test_airplane])

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    # noise_fn is the function which will sample the latent vector from the gaussian distribution in this case
    noise_fn = lambda x: torch.normal(mean=0, std=1, size=(x, latent_dim), device=device)
    gan = DCGAN(latent_dim, noise_fn, dataloader, batch_size=batch_size, device=device)

    start = time()
    for i in range(epochs):
        print(f"Epoch {i+1}; Elapsed time = {int(time() - start)}s")

        g_loss, (d_loss_real, d_loss_fake) = gan.train_epoch()

        print(f"G_loss -> {g_loss}, D_loss_real -> {d_loss_real}, D_loss_fake -> {d_loss_fake}\n")

        # save grid of 64 imgs for each epoch, to see generator progress
        images = gan.generate_samples(num=64)
        ims = vutils.make_grid(images, normalize=True)
        plt.axis("off")
        plt.title(f"Epoch {i+1}")
        plt.imshow(np.transpose(ims, (1, 2, 0)))
        plt.savefig(f'dcgan_test_pytorch/epoch{i+1}.png')

