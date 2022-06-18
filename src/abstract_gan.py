from abc import ABC, abstractmethod
import shutil
import os

import random
import re
import pandas as pd
from time import time
from tqdm import tqdm
from typing import List

import matplotlib.pyplot as plt

import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import ConcatDataset, DataLoader

from src.utils import clean_dataset, PaintingsFolder, device, ClasslessImageFolder
from src.metrics import Metric, GANMetricFake, GANMetricRealFake

class GAN(ABC):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    @staticmethod
    def get_transformers(image_size, normalization_values = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), data_augmentation: bool = False):
        if data_augmentation:
            return transforms.Compose([transforms.TrivialAugmentWide(),
                                       transforms.Resize((image_size, image_size)),
                                       transforms.CenterCrop((image_size, image_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(normalization_values[0], normalization_values[1])])
        else:
            return transforms.Compose([transforms.Resize(image_size), 
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(normalization_values[0], normalization_values[1])])
    
    @classmethod
    def prepare_dataset(cls, image_size: int, normalization_values = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):

        dataset_path='../dataset/best_artworks/resized/resized' 
        metadata_path='../dataset/best_artworks/artists.csv'
        metadata_csv = pd.read_csv(metadata_path)

        death_monet = 1926
        impressionist_artists_dict = dict()
        other_artists_to_consider_dict = dict()
        for artist_id, artist_name, artist_years, artistic_movement in zip(metadata_csv['id'],
                                                                           metadata_csv['name'],
                                                                           metadata_csv['years'],
                                                                           metadata_csv['genre']):

            dob = artist_years.split(' ')[0]
            if re.search(r'impressionism', artistic_movement.lower()):

                impressionist_artists_dict[artist_name] = artist_id
            elif int(dob) < death_monet:
                other_artists_to_consider_dict[artist_name] = artist_id

        clean_dataset(dataset_path)

        train_impressionist = PaintingsFolder(
            root=dataset_path,
            transform=cls.get_transformers(image_size=image_size,
                                           normalization_values=normalization_values),
            artists_dict=impressionist_artists_dict
        )

        train_impressionist_augment1 = PaintingsFolder(
            root=dataset_path,
            transform=cls.get_transformers(image_size=image_size,
                                           normalization_values=normalization_values, 
                                           data_augmentation=True),
            artists_dict=impressionist_artists_dict
        )

        train_others = PaintingsFolder(
            root=dataset_path,
            transform=cls.get_transformers(image_size=image_size,
                                           normalization_values=normalization_values),
            artists_dict=other_artists_to_consider_dict
        )

        # Ugly for loop but better for efficiency, we save 5 random paintings for each 'other artist'.
        # we exploit the fact that first we have all paintings of artist x, then we have all paintings of artist y, etc.
        subset_idx_list = []
        current_artist_id = train_others.targets[0]
        idx_paintings_current = []
        for i in range(len(train_others.targets)):
            if train_others.targets[i] != current_artist_id:
                idx_paintings_to_hold = random.sample(idx_paintings_current, 5)
                subset_idx_list.extend(idx_paintings_to_hold)

                current_artist_id = train_others.targets[i]
                idx_paintings_current.clear()
            else:
                indices_paintings = idx_paintings_current.append(i)

        train_others = torch.utils.data.Subset(train_others, subset_idx_list)

        # concat original impressionist, augmented impressionist, 5 random paintings of other artists
        dataset = ConcatDataset([train_impressionist, train_impressionist_augment1, train_others])

        return dataset
    
    def plot_training_images(self, images, i):
        ims = vutils.make_grid(images, normalize=True)
        return wandb.Image(ims, caption=str("TRAINING IMAGES GENERATED AT EPOCH: " + str(i+1)))
    
    def setup_directories(self, save_model_checkpoints):
        shutil.rmtree(str("output/" + self.__class__.__name__), ignore_errors=True)
        os.makedirs(str("output/" + self.__class__.__name__))

        if save_model_checkpoints:
            os.makedirs(str("output/" + self.__class__.__name__ + "/checkpoints"))
    
    @abstractmethod
    def save_model(self, path: str):
        raise NotImplementedError
    
    @abstractmethod
    def load_model(self, path: str, for_inference: bool):
        raise NotImplementedError
    
    @abstractmethod
    def set_train_mode(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_eval_mode(self):
        raise NotImplementedError
    

class Latent_GAN(GAN):

    def __init__(self, latent_dim: int,
                generator: nn.Module = None, discriminator: nn.Module = None,
                optim_g: torch.optim = None, optim_d: torch.optim = None,
                noise_fn = None) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.noise_fn = noise_fn if noise_fn is not None else lambda x: torch.normal(mean=0, std=1, size=(x, latent_dim, 1, 1), device=device)

        self.generator = generator
        self.discriminator = discriminator

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        self.optim_g = optim_g
        self.optim_d = optim_d

    def generate_samples(self, num=None, **args):
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
        try:
            latent_vec = args["latent_vec"]
        except KeyError:
            if num is None:
                raise ValueError("Must provide either a number of samples or a latent_vec in method generate_samples")
            
            latent_vec = self.noise_fn(num)
        
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.detach().cpu()  # move images to cpu
        return samples
    
    def save_model(self, path: str):
        torch.save(self.generator, path+"/generator.pt")
        torch.save(self.discriminator, path+"/discriminator.pt")
        torch.save(self.optim_g, path+"/generator_optimizer.pt")
        torch.save(self.optim_d, path+"/discriminator_optimizer.pt")
    
    def load_model(self, path: str, for_inference: bool):
        self.generator = torch.load(path+"/generator.pt")
        self.discriminator = torch.load(path+"/discriminator.pt")

        if for_inference:
            self.set_eval_mode()
            del self.optim_g
            self.optim_g = None
            del self.optim_d
            self.optim_d = None
        else:
            self.set_train_mode()
            self.optim_g = torch.load(path+"/generator_optimizer.pt")
            self.optim_d = torch.load(path+"/discriminator_optimizer.pt")
    
    def set_train_mode(self):
        self.generator.train()
        self.discriminator.train()
    
    def set_eval_mode(self):
        self.generator.eval()
        self.discriminator.eval()
    
    def train(self, batch_size: int, image_size: int, epochs: int, 
              save_model_checkpoints: bool = False, wandb_plot: bool = False,
              normalization_values = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
        
        if wandb_plot:
            run = wandb.init(project="Painter GAN", entity="painter_gan",
                             name="{gan_name}_E:{num_epochs}_B:{batch_size}_I:{image_size}".format(
                                gan_name=str(self.__class__.__name__).lower(), num_epochs=str(epochs), batch_size=str(batch_size), image_size=str(image_size)),
                             config={"epochs": epochs, "batch_size": batch_size})
            wandb.define_metric("epoch")

        data = self.prepare_dataset(image_size, normalization_values)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

        start = time()

        self.setup_directories(save_model_checkpoints)

        if wandb_plot:
            static_noise = self.noise_fn(batch_size)
        
        metrics_defined_in_wandb = False

        for i in range(epochs):
            print(f"Epoch {i+1}; Elapsed time = {int(time() - start)}s")

            losses = self.train_epoch(dataloader)

            if wandb_plot:
                if not metrics_defined_in_wandb:
                    for loss_name in losses.keys():
                        wandb.define_metric(loss_name, step_metric="epoch")
                    metrics_defined_in_wandb = True

            num_of_losses = len(losses)

            for j, (loss_name, loss_value) in enumerate(losses.items()):
                if j != num_of_losses-1:
                    end = ', '
                else:
                    end = '.'
                
                print(loss_name + "= " + str(loss_value), end=end)

            if (i+1)%100 == 0 and save_model_checkpoints:
                os.makedirs(str("output/" + self.__class__.__name__ + "/checkpoints/"+str(i+1)))
                self.save_model(str("output/" + self.__class__.__name__ + "/checkpoints/"+str(i+1)))
            
            if wandb_plot:
                log_dict = {"epoch": i+1}
                for loss_name, loss_value in losses.items():
                    log_dict[loss_name] = loss_value
                if ((i+1)%10 == 0 or (i == 0)):
                    images = self.generate_samples(batch_size, latent_vec=static_noise)
                    log_dict['training_images'] = self.plot_training_images(images, i)
                wandb.log(log_dict)
            
            print()
        
        if wandb_plot:
            run.finish()
    
    def train_epoch(self, dataloader):
        """Train both networks for one epoch and return the losses."""
        running_losses = {}

        n_batches = len(dataloader)
        for (real_samples, _) in tqdm(dataloader, total=n_batches):
            real_samples = real_samples.to(device)

            batch_losses = self.train_step(real_samples)

            if len(running_losses) == 0:
                running_losses = dict.fromkeys(list(batch_losses.keys()), 0)

            for loss_name, loss_value in batch_losses.items():
                running_losses[loss_name] += loss_value

        for loss_name, loss_value in running_losses.items():
            running_losses[loss_name] = loss_value / n_batches

        return running_losses
    
    @abstractmethod
    def train_step(self, real_data):
        raise NotImplementedError
    
    def test(self, real_data, metrics_to_consider: List[Metric]):
        real_images = []
        print("STARTING TESTING")
        print("LOADING REAL DATA")
        for (real, _) in tqdm(real_data):
            real_images.append(real)
        real_images = torch.stack(real_images)

        size = real_data.size(0)
        fake_images = self.generate_samples(size)

        for metric in metrics_to_consider:
            if isinstance(metric, GANMetricFake):
                metric.update(fake_images)
            elif isinstance(metric, GANMetricRealFake):
                metric.update(real_images, real=True)
                metric.update(fake_images, real=False)
        
        results = {}
        for metric in metrics_to_consider:
            results[str(metric)] = metric.compute()
            metric.reset()
        
        return results

class AB_GAN(GAN):

    def __init__(self, generator_a2b: nn.Module = None, generator_b2a: nn.Module = None,
                discriminator_a: nn.Module = None, discriminator_b: nn.Module = None,
                optim_g: torch.optim = None, optim_d: torch.optim = None,
                lr_scheduler_class: type = None) -> None:
        super().__init__()

        self.generator_a2b = generator_a2b
        self.generator_b2a = generator_b2a
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b

        self.generator_a2b.apply(self.weights_init)
        self.generator_b2a.apply(self.weights_init)
        self.discriminator_a.apply(self.weights_init)
        self.discriminator_b.apply(self.weights_init)  

        self.optim_g = optim_g
        self.optim_d = optim_d

        self.lr_scheduler_class = lr_scheduler_class

        self.lr_scheduler_g = None
        self.lr_scheduler_d = None
    
    @classmethod
    def load_dataset(cls, dataset_path, transformers):
        if all([os.path.isdir(os.path.join(dataset_path, name)) for name in os.listdir(dataset_path)]):
            return datasets.ImageFolder(root=dataset_path, transform=transformers)
        elif all([os.path.isfile(os.path.join(dataset_path, name)) for name in os.listdir(dataset_path)]):
            return ClasslessImageFolder(root=dataset_path, transform=transformers)
        else:
            raise FileNotFoundError("Specified dataset path " + dataset_path + " must contain only sub-directories or files")

    @classmethod
    def prepare_dataset_b(cls, image_size: int, normalization_values = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          dataset_path: str = "../dataset/photo_jpg",):
        transformers=cls.get_transformers(image_size=image_size, normalization_values=normalization_values)

        return cls.load_dataset(dataset_path, transformers) 
    
    def generate_images_a2b(self, images_vec=None):
        """
        Given a list of images from domain A, passes it to the generator 
        to obtain images associated to the domain B
        """
        with torch.no_grad():
            return self.generator_a2b(images_vec).detach().cpu()
    
    def generate_images_b2a(self, images_vec=None):
        """
        Given a list of images from domain B, passes it to the generator 
        to obtain images associated to the domain A
        """
        with torch.no_grad():
            return self.generator_b2a(images_vec).detach().cpu()
    
    def save_model(self, path: str):
        torch.save(self.generator_a2b, path+"/generator_a2b.pt")
        torch.save(self.generator_b2a, path+"/generator_b2a.pt")
        torch.save(self.discriminator_a, path+"/discriminator_a.pt")
        torch.save(self.discriminator_b, path+"/discriminator_b.pt")
        torch.save(self.optim_g, path+"/generators_optimizer.pt")
        torch.save(self.optim_d, path+"/discriminators_optimizer.pt")
        if self.lr_scheduler_g is not None:
            torch.save(self.lr_scheduler_g, path+"/generator_scheduler.pt")
            torch.save(self.lr_scheduler_d, path+"/discriminator_scheduler.pt")
    
    def load_model(self, path: str, for_inference: bool):
        self.generator_a2b = torch.load(path+"/generator_a2b.pt")
        self.generator_b2a = torch.load(path+"/generator_b2a.pt")
        self.discriminator_a = torch.load(path+"/discriminator_a.pt")
        self.discriminator_b = torch.load(path+"/discriminator_b.pt")

        if for_inference:
            self.set_eval_mode()
            del self.optim_g
            self.optim_g = None
            del self.optim_d
            self.optim_d = None
            if True:
                del self.lr_scheduler_g
                self.lr_scheduler_g = None
                del self.lr_scheduler_d
                self.lr_scheduler_d = None
        else:
            self.set_train_mode()
            self.optim_g = torch.load(path+"/generators_optimizer.pt")
            self.optim_d = torch.load(path+"/discriminators_optimizer.pt")
            if os.path.isfile(path+"/generator_scheduler.pt"):
                self.lr_scheduler_g = torch.load(path+"/generator_scheduler.pt")
                self.lr_scheduler_d = torch.load(path+"/discriminator_scheduler.pt")
    
    def set_train_mode(self):
        self.generator_a2b.train()
        self.generator_b2a.train()
        self.discriminator_a.train()
        self.discriminator_b.train()
    
    def set_eval_mode(self):
        self.generator_a2b.eval()
        self.generator_b2a.eval()
        self.discriminator_a.eval()
        self.discriminator_b.eval()
    
    def train(self, batch_size: int, image_size: int, epochs: int,
              save_model_checkpoints: bool = False, wandb_plot: bool = False,
              dataset_b_path = '../dataset/photo_jpg',
              dataset_b_progress = '../dataset/photo_jpg_test',
              normalization_values = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
              scheduler_params: dict = None):
        
        if wandb_plot:
            run = wandb.init(project="Painter GAN", entity="painter_gan",
                             name="{gan_name}_E:{num_epochs}_B:{batch_size}_I:{image_size}".format(
                                gan_name=str(self.__class__.__name__).lower(), num_epochs=str(epochs), batch_size=str(batch_size), image_size=str(image_size)),
                             config={"epochs": epochs, "batch_size": batch_size})
            wandb.define_metric("epoch")
            
        if scheduler_params != None:
            self.lr_scheduler_g = self.lr_scheduler_class(self.optim_g, **scheduler_params)
            self.lr_scheduler_d = self.lr_scheduler_class(self.optim_d, **scheduler_params)

        dataset_a = self.prepare_dataset(image_size, normalization_values)
        dataset_b = self.prepare_dataset_b(image_size, normalization_values, dataset_b_path)

        dataloader_a = DataLoader(dataset_a, batch_size=batch_size, shuffle=True, num_workers=0)
        dataloader_b = DataLoader(dataset_b, batch_size=batch_size, shuffle=True, num_workers=0)

        if wandb_plot:
            test_set = self.load_dataset(dataset_b_progress, self.get_transformers(image_size, normalization_values))

            test_set_list = []
            for i, (image, _) in enumerate(test_set):
                if i < batch_size:
                    test_set_list.append(image.to(device))
                else:
                    break
            test_set_list = torch.stack(test_set_list)

        start = time()

        self.setup_directories(save_model_checkpoints)

        metrics_defined_in_wandb = False

        for i in range(epochs):
            print(f"Epoch {i+1}; Elapsed time = {int(time() - start)}s")

            losses = self.train_epoch(dataloader_a, dataloader_b)

            if wandb_plot:
                if not metrics_defined_in_wandb:
                    for loss_name in losses.keys():
                        wandb.define_metric(loss_name, step_metric="epoch")
                    metrics_defined_in_wandb = True

            num_of_losses = len(losses)

            for j, (loss_name, loss_value) in enumerate(losses.items()):
                if j != num_of_losses-1:
                    end = ', '
                else:
                    end = '.'
                
                print(loss_name + "= " + str(loss_value), end=end)

            if wandb_plot:
                log_dict = {"epoch": i+1}
                for loss_name, loss_value in losses.items():
                    log_dict[loss_name] = loss_value
                if ((i+1)%10 == 0 or (i == 0)):
                    images = self.generate_images_b2a(test_set_list)
                    log_dict['training_images'] = self.plot_training_images(images, i)
                wandb.log(log_dict)

            if (i+1)%100 == 0 and save_model_checkpoints:
                os.makedirs(str("output/" + self.__class__.__name__ + "/checkpoints/"+str(i+1)))
                self.save_model(str("output/" + self.__class__.__name__ + "/checkpoints/"+str(i+1)))
            
            print()
        
        if wandb_plot:
            run.finish()
    
    def train_epoch(self, dataloader_a, dataloader_b):

        running_losses = {}

        n_batches = min(len(dataloader_a), len(dataloader_b))
        for (images_a, images_b) in tqdm(zip(dataloader_a, dataloader_b), total=n_batches):
            
            real_A = images_a[0].to(device)
            real_B = images_b[0].to(device)

            batch_losses = self.train_step(real_A, real_B)

            if len(running_losses) == 0:
                running_losses = dict.fromkeys(list(batch_losses.keys()), 0)

            for loss_name, loss_value in batch_losses.items():
                running_losses[loss_name] += loss_value

        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.step()
            self.lr_scheduler_d.step()

        for loss_name, loss_value in running_losses.items():
            running_losses[loss_name] = loss_value / n_batches

        return running_losses
    
    @abstractmethod
    def train_step(self, real_data_a, real_data_b):
        raise NotImplementedError
    
    def test(self, real_data, test_data, metrics_to_consider: List[Metric]):
        real_images = []
        print("STARTING TESTING")
        print("LOADING REAL DATA")
        for (real, _) in tqdm(real_data):
            real_images.append(real)
        real_images = torch.stack(real_images)
        real_images = real_images[:64]

        fake_images = []
        print("LOADING FAKE DATA")
        for (fake, _) in tqdm(test_data):
            fake_images.append(fake.to(device))
        fake_images = torch.stack(fake_images)
        fake_images = self.generate_images_b2a(fake_images)

        for metric in metrics_to_consider:
            if isinstance(metric, GANMetricFake):
                metric.update(fake_images)
            elif isinstance(metric, GANMetricRealFake):
                metric.update(real_images, real=True)
                metric.update(fake_images, real=False)
        
        results = {}
        for metric in metrics_to_consider:
            results[str(metric)] = metric.compute()
            metric.reset()
        
        return results