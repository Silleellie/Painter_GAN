# Painter_GAN

Various GANs architectures and strategies compared to tackle the [I'm something of a painter myself](https://www.kaggle.com/c/gan-getting-started) kaggle competition

# General Information

All the source code for each GAN architecture is contained in the `src` folder.
* Each file (one for every GAN architecture) contains a `main` method in which there's example on how to perform the train of the related gan

**Please note**: The plots and images generated during training, as well as the results for the test phase can be found [here](https://wandb.ai/painter_gan/Painter%20GAN)

# Usage

## Train with default augmented "best artworks" dataset

In order to perform training of a specific GAN architecture with the augmented dataset strategy as described in the [attached paper](GANs_for_Monet_Paintings.pdf), download the following [kaggle dataset](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) and unpack it in the `dataset/best_artworks` folder.

Then simply instantiate the GAN and call the `train_with_default_dataset()` method.

So, if this is `train_script.py` placed at the root of the repository:

```python
from src.dcgan import DCGAN

gan = DCGAN()

gan.train_with_default_dataset(batch_size=128,
                               image_size=64,
                               epochs=5,
                               save_model_checkpoints=True,
                               save_imgs_local=True,
                               wandb_plot=False)
```

Your folder structure should be like the following:

```
📁 dataset/
├── 📁 best_artworks/
│   └── 📁 images/
│   └── 📁 resized/
│   └── 📄 artists.csv
📁 src
📄 README.md
📄 .gitignore
📄 train_script.py
```

## Train with custom dataset

If you wish to perform training of a particular GAN with a custom dataset, then simply instantiate the GAN and call the `train_with_custom_dataset()` method. In this case no preprocessing/data augmentation is applied by default and you need to pass the path of the folder containing train images:

```python
from src.dcgan import DCGAN

gan = DCGAN()

gan.train_with_custom_dataset(dataset_path,
                              batch_size=128,
                              image_size=64,
                              epochs=5,
                              save_model_checkpoints=True,
                              save_imgs_local=True,
                              wandb_plot=False)
```

## Train SyleGAN3

StyleGAN3 has a different abstraction w.r.t all the other implemented GANs, since it exploits Transfer Learning. To perform training simply instantiate the `StyleGAN3` class and call its `prepare_dataset()` method followed by the `train()` method:

```python
from src import StyleGAN3

gan = StyleGAN3()

gan.prepare_dataset(original_dataset_path, output_prepared_path)
gan.train(output_prepared_path, resume_from=checkpoint_path, output_dir=output_folder_path)
```

## Generate Samples

One you have trained your model, you can generate and save images via the following methods:

```python
# for latent GANs (DCGAN, BEGAN, etc.)
gan.save_generated_images(output_path, sample_size=100)

# for unpaired image to image models (CycleGAN, DiscoGAN)
gan.save_generated_images(domain_a_imgs_path, output_path, sample_size=100)

# for StyleGAN3
gan.generate_images(model_to_load_path, output_dir, seeds_interval=(1,100))
```

## Evaluate generated images

In order to evaluate generated samples with FID and MiFID metric simply call the `plot_fid_mifid` function specifying the path containing the generated images and path containing real images

```python
from src.fid_mifid import plot_fid_mifid

plot_fid_mifid(generated_images_path, real_images_path)
```

Fid and MiFID will be printed out
