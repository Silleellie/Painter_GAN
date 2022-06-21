from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from torchmetrics import Metric
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance, NoTrainInceptionV3, _compute_fid
from torchmetrics.utilities.data import dim_zero_cat
from torch.nn.functional import cosine_similarity

import wandb


# implementation of the MiFid metric based on the original implementation of the fid metric in the torchmetrics library
# it expands the Fid implementation by adding the computations associated to the MiFid metric
from torchvision.transforms import transforms
from tqdm import tqdm

from src.utils import ClasslessImageFolder, device


class MiFID(Metric):
    def __init__(self, feature=2048, epsilon=1e-6, **args):
        super().__init__(**args)
        self.epsilon = epsilon

        self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        self.add_state("d", default=[], dist_reduce_fx=None)
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, images, real: bool):
        features = self.inception(images)
        self.orig_d_type = features.dtype

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)
    
    def compute(self):
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)

        orig_dtype = real_features.dtype

        real_features = real_features.double()
        fake_features = fake_features.double()

        n = real_features.shape[0]
        m = fake_features.shape[0]
        mean1 = real_features.mean(dim=0)
        mean2 = fake_features.mean(dim=0)
        diff1 = real_features - mean1
        diff2 = fake_features - mean2
        cov1 = 1.0 / (n - 1) * diff1.t().mm(diff1)
        cov2 = 1.0 / (m - 1) * diff2.t().mm(diff2)

        fid_value = _compute_fid(mean1, cov1, mean2, cov2).to(orig_dtype)

        # computes the memorization distance (key value for the MiFid metric)
        # finds the minimum cosine similarity between each generated feature vector and the real ones
        # also applies a threshold epsilon and this distance is returned only if it surpasses said threshold,
        # otherwise equal to 1

        min_distances_vector = []

        for fake_feature_vector in fake_features:
            distances_vector = []
            for real_feature_vector in real_features:
                cos_distance = 1 - cosine_similarity(real_feature_vector, fake_feature_vector, dim=0)
                distances_vector.append(cos_distance)
            min_distances_vector.append(min(distances_vector))
        
        d = np.mean(min_distances_vector)
        d = d if d < self.epsilon else 1

        mifid = fid_value * (1 / d)

        return mifid


class GANMetric(ABC):

    def __init__(self, torch_metric: Metric) -> None:
        self.torch_metric = torch_metric
    
    @abstractmethod
    def update(self, images):
        raise NotImplementedError
    
    def compute(self):
        return self.torch_metric.compute()
    
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
    
    def reset(self):
        self.torch_metric.reset()


# A distinction has been made between two categories of GAN metrics
# - Those that work by only using the generated images [GANMetricFake] (Inception Score, for example)
# - Those that work by using both the real images and the generated ones [GANMetricRealFake] (Frechet Inception
# Distance, for example)

class GANMetricFake(GANMetric):

    def __init__(self, torch_metric: Metric) -> None:
        super().__init__(torch_metric)
    
    def update(self, images):
        if images.type() != torch.uint8:
            self.torch_metric.update(images.type(torch.uint8))
        else:
            self.torch_metric.update(images)


class GANMetricRealFake(GANMetric):

    def __init__(self, torch_metric: Metric) -> None:
        super().__init__(torch_metric)
    
    def update(self, images, real: bool):
        if images.type() != torch.uint8:
            self.torch_metric.update(images.type(torch.uint8), real=real)
        else:
            self.torch_metric.update(images, real=real)


# Classes of the actual metrics follow
# args in the init method is used to pass to the actual torch metric possible parameters

class KIDMetric(GANMetricRealFake):

    def __init__(self, **args) -> None:
        super().__init__(KernelInceptionDistance(**args))
    
    def __str__(self) -> str:
        return "Kernel Inception Distance (KID)"


class FIDMetric(GANMetricRealFake):

    def __init__(self, **args) -> None:
        super().__init__(FrechetInceptionDistance(**args))
    
    def __str__(self) -> str:
        return "Frechet Inception Distance (FID)"


class MIFIDMetric(GANMetricRealFake):

    def __init__(self, **args) -> None:
        super().__init__(MiFID(**args))
    
    def __str__(self) -> str:
        return "Memorization-Informed Frechet Inception Distance (MIFID)"


class ISMetric(GANMetricFake):

    def __init__(self, **args) -> None:
        super().__init__(InceptionScore(**args))
    
    def __str__(self) -> str:
        return "Inception Score (IS)"


class TestEvaluate:

    def __init__(self, path_fake: str, path_real: str = None, image_size: int = 64):

        transf = transforms.Compose([transforms.Resize((image_size, image_size)),
                                     transforms.PILToTensor()])

        fake_data = ClasslessImageFolder(path_fake, transform=transf)
        fake_images = []
        for (fake, _) in tqdm(fake_data, desc="Loading generated images"):
            fake_images.append(fake)
        self.fake_images = torch.stack(fake_images)

        self.real_images = None
        if path_real is not None:

            real_data = ClasslessImageFolder(path_real, transform=transf)

            real_images = []
            for (real, _) in tqdm(real_data, desc="Loading real images"):
                real_images.append(real)
            self.real_images = torch.stack(real_images)

    def perform(self, metrics: List[GANMetric], wandb_plot: bool = False, run_name: str = "test_run"):
        if any(isinstance(metric, GANMetricRealFake) for metric in metrics) and self.real_images is None:
            raise ValueError("You must pass also the path of the fake images!")
        
        if wandb_plot:
            run = wandb.init(project="Painter GAN", entity="painter_gan", name=run_name)
            for metric in metrics:
                wandb.define_metric(str(metric))

        for metric in metrics:
            print("COMPUTING METRIC: ", str(metric))
            if isinstance(metric, GANMetricFake):
                metric.update(self.fake_images)
            elif isinstance(metric, GANMetricRealFake):
                metric.update(self.real_images, real=True)
                metric.update(self.fake_images, real=False)

        results = {}
        for metric in metrics:
            results[str(metric)] = metric.compute()
            metric.reset()
        
        if wandb_plot:
            wandb.log(results)
            run.finish()

        return results
