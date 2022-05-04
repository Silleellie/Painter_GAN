import os
import shutil

import torch
import torch.nn as nn
from torch import optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, latent_dimension, num_filters):
        super(Decoder, self).__init__()
        self.num_filters = num_filters
        self.h = latent_dimension  # latent dimension is called 'h' in paper

        self._init_modules()

    def _init_modules(self):
        self.elu = nn.ELU(inplace=True)
        self.tanh = nn.Tanh()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

        self.h0 = nn.Linear(self.h, self.num_filters * 8 * 8)
        self.conv1 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)

        self.conv3 = nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1)
        self.conv4 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)

        self.conv5 = nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1)
        self.conv6 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)

        self.conv7 = nn.Conv2d(self.num_filters, 3, 3, 1, 1)

    def forward(self, input):
        h0 = self.h0(input)

        # reshape
        # -1 means to infer batch size from input passed
        h0 = h0.view((-1, self.num_filters, 8, 8))

        x = self.elu(self.conv1(h0))
        x = self.elu(self.conv2(x))
        x = self.up(x)

        # upsample and inject h0
        upsampled_h = self.up(h0)
        x = torch.cat([x, upsampled_h], dim=1)

        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.up(x)

        # upsample and inject
        upsampled_h = self.up(upsampled_h)
        x = torch.cat([x, upsampled_h], dim=1)

        x = self.elu(self.conv5(x))
        x = self.elu(self.conv6(x))

        x = self.elu(self.conv7(x))
        x = self.tanh(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dimension, num_filters):
        super(Encoder, self).__init__()
        self.num_filters = num_filters
        self.h = latent_dimension

        self._init_modules()

    def _init_modules(self):
        self.elu = nn.ELU(inplace=True)

        # last conv of each layer will take care of subsampling with stride=2

        self.conv0 = nn.Conv2d(3, self.num_filters, 3, 1, 1)
        self.conv1 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.num_filters, 2 * self.num_filters, 3, 2, 1)

        self.conv3 = nn.Conv2d(2 * self.num_filters, 2 * self.num_filters, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * self.num_filters, 3 * self.num_filters, 3, 2, 1)

        self.conv5 = nn.Conv2d(3 * self.num_filters, 3 * self.num_filters, 3, 1, 1)
        self.conv6 = nn.Conv2d(3 * self.num_filters, 3 * self.num_filters, 3, 1, 1)

        # output of encoder will be input of encoder which expects input of size 'h'
        self.fc = nn.Linear(8 * 8 * 3 * self.num_filters, self.h)

    def forward(self, input):
        x = self.elu(self.conv0(input))
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))

        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))

        x = self.elu(self.conv5(x))
        x = self.elu(self.conv6(x))

        # reshape
        # -1 means to infer batch size from input passed
        x = x.view((-1, 3 * self.num_filters * 8 * 8))
        x = self.fc(x)

        return x


# alias for the decoder, since generator has same architecture
Generator = Decoder


class Discriminator(nn.Module):
    def __init__(self, latent_dimension=100, num_filters=64):
        super(Discriminator, self).__init__()
        self.enc = Encoder(latent_dimension=latent_dimension, num_filters=num_filters)
        self.dec = Decoder(latent_dimension=latent_dimension, num_filters=num_filters)

    def forward(self, input):
        return self.dec(self.enc(input))


class BEGAN:
    def __init__(self, latent_dim, num_filters, noise_fn, dataloader,
                 batch_size=32, lr_d=0.0004, lr_g=0.0004, device: torch.device = 'cpu'):
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
        self.generator = Generator(latent_dim, num_filters).to(device)
        self.generator.apply(weights_init)

        self.discriminator = Discriminator(latent_dim, num_filters).to(device)
        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.device = device

        self.criterion = nn.L1Loss()
        self.optim_d = optim.Adam(self.discriminator.parameters(),
                                  lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.999))
        self.gamma = 0.5
        self.lambda_k = 0.001
        self.Kt = 0.0

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

        latent_vec = self.noise_fn(current_batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(generated, classifications)  # l1_loss

        self.generator.zero_grad()
        loss.backward()
        self.optim_g.step()

        return loss

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, real_samples)  # l1_loss

        # generated samples
        latent_vec = self.noise_fn(current_batch_size)
        fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples.detach())
        loss_fake = self.criterion(pred_fake, fake_samples)  # l1_loss

        # combine two losses
        # Simple sum and no division by two seems to work better, check original dcgan paper though
        loss = loss_real - self.Kt * loss_fake

        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()

        return loss, loss_real, loss_fake

    def train_epoch(self):
        loss_g_running, loss_d_running, M_running = 0, 0, 0
        for batch, (real_samples, _) in enumerate(self.dataloader):
            current_batch_size = real_samples.size(0)

            real_samples = real_samples.to(self.device)

            # ----------
            #  Training generator
            # ----------
            loss_g = self.train_step_generator(current_batch_size)

            # ----------
            #  Training discriminator
            # ----------
            loss_d, loss_d_real, loss_d_fake = self.train_step_discriminator(real_samples, current_batch_size)

            # ----------------
            # Update weights
            # ----------------
            # Kt update
            balance = (self.gamma * loss_d_real - loss_d_fake)
            self.Kt = max(min(self.Kt + self.lambda_k * balance.item(), 1.0), 0.0)

            # Update convergence metric
            M = (loss_d_real + torch.abs(balance)).item()

            # ----------------
            # Update metrics for logging
            # ----------------
            loss_d_running += loss_d.item()
            loss_g_running += loss_g.item()
            M_running += M

        n_batches = len(self.dataloader)
        loss_g_running /= n_batches
        loss_d_running /= n_batches
        M_running /= n_batches

        return loss_g_running, loss_d_running, M_running, self.Kt


if __name__ == '__main__':
    """
    BEGAN PAPER
    https://arxiv.org/pdf/1703.10717.pdf
    
    IMPROVED BEGAN PAPER
    https://wlouyang.github.io/Papers/iBegan.pdf
    """
    output_dir = "../output/began_test_pytorch"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    image_size = 32
    batch_size = 32
    epochs = 200
    latent_dim = 128
    num_filters = 32

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

    # slow learning rate since with 0.0004 it missed the minimum generating black images, but maybe it was an
    # accidental thing
    gan = BEGAN(latent_dim, num_filters, noise_fn, dataloader, batch_size=batch_size, device=device,
                lr_d=2e-4, lr_g=2e-4)

    print("Started training...")
    for i in range(epochs):
        g_loss, d_loss, M, Kt = gan.train_epoch()

        print(
            "[Epoch %d/%d] [G loss: %f] [D loss: %f]  -- M: %f, k: %f"
            % (i + 1, epochs, g_loss, d_loss, M, Kt)
        )

        # save grid of 64 imgs for each epoch, to see generator progress
        images = gan.generate_samples(num=64)
        ims = vutils.make_grid(images, normalize=True)
        plt.axis("off")
        plt.title(f"Epoch {i + 1}")
        plt.imshow(np.transpose(ims, (1, 2, 0)))
        plt.savefig(f'{output_dir}/epoch{i + 1}.png')
