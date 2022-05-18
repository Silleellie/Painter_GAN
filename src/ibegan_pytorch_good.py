import os
import shutil

import numpy as np
import torch
from torch import optim

from began_pytorch_good import BEGAN
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dset
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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
        self.batchnorm0 = nn.BatchNorm2d(self.num_filters)

        self.conv0 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
        self.conv1 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
        self.batchnorm1 = nn.BatchNorm2d(self.num_filters)

        self.conv2 = nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1)
        self.conv3 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
        self.batchnorm2 = nn.BatchNorm2d(self.num_filters)

        self.conv4 = nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1)
        self.conv5 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
        self.batchnorm3 = nn.BatchNorm2d(self.num_filters)

        self.conv6 = nn.Conv2d(self.num_filters, 3, 3, 1, 1)

    def forward(self, input):
        h0 = self.h0(input)

        # reshape
        # -1 means to infer batch size from input passed
        h0 = h0.view((-1, self.num_filters, 8, 8))
        h0 = self.batchnorm0(h0)

        x = self.elu(self.conv0(h0))
        x = self.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.up(x)

        # upsample and inject h0
        upsampled_h = self.up(h0)
        x = torch.cat([x, upsampled_h], dim=1)

        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.batchnorm2(x)
        x = self.up(x)

        # upsample and inject h0
        upsampled_h = self.up(upsampled_h)
        x = torch.cat([x, upsampled_h], dim=1)

        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))
        x = self.batchnorm3(x)

        x = self.elu(self.conv6(x))
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
        self.batchnorm0 = nn.BatchNorm2d(self.num_filters)

        self.conv1 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.num_filters, 2 * self.num_filters, 3, 2, 1)
        self.batchnorm1 = nn.BatchNorm2d(2 * self.num_filters)

        self.conv3 = nn.Conv2d(2 * self.num_filters, 2 * self.num_filters, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * self.num_filters, 3 * self.num_filters, 3, 2, 1)
        self.batchnorm2 = nn.BatchNorm2d(3 * self.num_filters)

        self.conv5 = nn.Conv2d(3 * self.num_filters, 3 * self.num_filters, 3, 1, 1)
        self.conv6 = nn.Conv2d(3 * self.num_filters, 3 * self.num_filters, 3, 1, 1)
        self.batchnorm3 = nn.BatchNorm2d(3 * self.num_filters)

        # output of encoder will be input of encoder which expects input of size 'h'
        self.fc = nn.Linear(8 * 8 * 3 * self.num_filters, self.h)

    def forward(self, input):
        x = self.elu(self.conv0(input))
        x = self.batchnorm0(x)

        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.batchnorm1(x)

        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.batchnorm2(x)

        x = self.elu(self.conv5(x))
        x = self.elu(self.conv6(x))
        x = self.batchnorm3(x)

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


class IBEGAN(BEGAN):

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
        super(IBEGAN, self).__init__(latent_dim, num_filters, noise_fn, dataloader, batch_size,
                                     lr_d, lr_g, device)

        self.generator = Generator(latent_dim, num_filters).to(device)
        self.generator.apply(weights_init)

        self.discriminator = Discriminator(latent_dim, num_filters).to(device)
        self.discriminator.apply(weights_init)

        # need to redefine optimizer since we changed generator and discriminator
        self.optim_d = optim.Adam(self.discriminator.parameters(),
                                  lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.999))

        self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(self.optim_g, factor=0.5)
        self.scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(self.optim_d, factor=0.5)

        self.criterion_noisy = nn.MSELoss()
        self.lambda_noise = 2

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, real_samples)  # l1_loss

        # generated samples
        latent_vec = self.noise_fn(current_batch_size)
        fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples.detach())
        loss_fake = self.criterion(pred_fake, fake_samples)  # l1_loss

        # denoising loss
        noise = torch.randn_like(real_samples)
        noisy_samples = real_samples + noise  # 0.3 to reduce noisy factor
        pred_noisy = self.discriminator(noisy_samples)
        loss_noisy = self.criterion_noisy(pred_noisy, noisy_samples)

        # ----------decomment to see noisy samples-------------
        # ims = vutils.make_grid(noisy_samples.cpu(), normalize=True)
        # plt.axis("off")
        # plt.title(f"Noisy samples")
        # plt.imshow(np.transpose(ims, (1, 2, 0)))
        # plt.savefig(f'noisy samples.png')

        # combine losses
        loss = loss_real - self.Kt * loss_fake + self.lambda_noise * loss_noisy

        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()

        return loss, loss_real, loss_fake


if __name__ == '__main__':
    """
    IMPROVED BEGAN PAPER
    https://wlouyang.github.io/Papers/iBegan.pdf
    """
    output_dir = "../output/ibegan_test_pytorch"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    image_size = 32
    batch_size = 128
    epochs = 1000
    latent_dim = 128
    num_filters = 64

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

    gan = IBEGAN(latent_dim, num_filters, noise_fn, dataloader, batch_size=batch_size, device=device,
                 lr_d=1e-4, lr_g=1e-4)

    print("Started training...")
    for i in range(epochs):
        g_loss, d_loss, M, Kt = gan.train_epoch()

        print(
            "[Epoch %d/%d] [G loss: %f] [D loss: %f]  -- M: %f, k: %f"
            % (i + 1, epochs, g_loss, d_loss, M, Kt)
        )

        gan.scheduler_g.step(g_loss)
        gan.scheduler_d.step(d_loss)

        # save grid of 64 imgs for each epoch, to see generator progress
        images = gan.generate_samples(num=64)
        ims = vutils.make_grid(images, normalize=True)
        plt.axis("off")
        plt.title(f"Epoch {i + 1}")
        plt.imshow(np.transpose(ims, (1, 2, 0)))
        plt.savefig(f'{output_dir}/epoch{i + 1}.png')
