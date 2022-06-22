import wandb
from pytorch_fid.fid_score import get_activations, InceptionV3, IMAGE_EXTENSIONS, calculate_frechet_distance, \
    calculate_fid_given_paths
from scipy.spatial.distance import cosine

import os
import numpy as np
import pathlib
from tqdm import tqdm


def calculate_mifid_given_paths(path_fake, path_real, batch_size, device, dims, num_workers=1, epsilon=1e-6):
    """Calculates the FID of two paths"""
    if not os.path.exists(path_fake):
        raise RuntimeError('Invalid path: %s' % path_fake)
    if not os.path.exists(path_real):
        raise RuntimeError('Invalid path: %s' % path_real)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    path_fake = pathlib.Path(path_fake)
    files_fake = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path_fake.glob('*.{}'.format(ext))])

    act_fake = get_activations(files_fake, model, batch_size, dims, device, num_workers)

    path_real = pathlib.Path(path_real)
    files_real = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path_real.glob('*.{}'.format(ext))])
    
    act_real = get_activations(files_real, model, batch_size, dims, device, num_workers)

    m1 = np.mean(act_fake, axis=0)
    s1 = np.cov(act_fake, rowvar=False)

    m2 = np.mean(act_real, axis=0)
    s2 = np.cov(act_real, rowvar=False)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    # computes the memorization distance (key value for the MiFid metric)
    # finds the minimum cosine similarity between each generated feature vector and the real ones
    # also applies a threshold epsilon and this distance is returned only if it surpasses said threshold,
    # otherwise equal to 1

    min_distances_vector = []

    for fake_feature_vector in tqdm(act_fake):
        distances_vector = []
        for real_feature_vector in act_real:
            cos_distance = cosine(real_feature_vector, fake_feature_vector)
            distances_vector.append(cos_distance)
        min_distances_vector.append(min(distances_vector))
    
    d = np.mean(min_distances_vector)
    d = d if d < epsilon else 1

    mifid = fid_value * (1 / d)

    return mifid


def plot_fid_mifid(path_fake: str, path_real: str, run_name: str = None):
    print("Calculating Fid Score...")

    fid = calculate_fid_given_paths([path_fake, path_real], batch_size=50, device='cuda:0', dims=2048)
    print("fid: ", fid)

    print("Calculating MiFid Score...")

    mifid = calculate_mifid_given_paths(path_fake=path_fake, path_real=path_real,
                                        batch_size=50,
                                        device='cuda:0',
                                        dims=2048,
                                        epsilon=10e-15)
    print("mifid: ", mifid)

    if run_name is not None:
        run = wandb.init(project='Painter GAN', entity='painter_gan', name=run_name)

        wandb.define_metric('FID')
        wandb.define_metric('MiFID')

        res_plot = {'FID': fid, 'MiFID': mifid}

        wandb.log(res_plot)

        run.finish()
