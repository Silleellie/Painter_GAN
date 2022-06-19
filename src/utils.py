import os
import re
import shutil
from pathlib import Path
from copy import deepcopy
from typing import Tuple, List, Dict, Callable, Optional
import torch
from PIL import Image

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def set_device(device: str):
    device = device

def clean_dataset(resized_images_dir):
    for filename in os.listdir(resized_images_dir):

        if os.path.isfile(os.path.join(resized_images_dir, filename)):
            if re.match(r'Albrecht_D(?=.+rer_(\d+))', filename):

                painting_number = re.findall(r'Albrecht_D(?=.+rer_(\d+)\.jpg)', filename)[0]

                old_fullpath = os.path.join(resized_images_dir, filename)

                filename = f'Albrecht_Dürer_{painting_number}.jpg'
                new_fullpath = os.path.join(resized_images_dir, filename)

                if not os.path.isfile(new_fullpath):
                    os.rename(old_fullpath, new_fullpath)

            artist_name = re.findall(r'(.*?[_.*?]*)(?=_\d+)', filename, re.UNICODE)[0]

            Path(os.path.join(resized_images_dir, artist_name)).mkdir(parents=True, exist_ok=True)
            shutil.move(os.path.join(resized_images_dir, filename), os.path.join(resized_images_dir,
                                                                                 artist_name,
                                                                                 filename))


class PaintingsFolder(ImageFolder):

    def __init__(
            self,
            root: str,
            artists_dict: Dict,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,

    ) -> None:

        # save artists in format 'name_surname'
        self.artists_dict = dict()
        for artist_name, artist_id in artists_dict.items():
            artist_names = artist_name.split(' ')

            valid_artist_name = '_'.join(artist_names)
            self.artists_dict[valid_artist_name] = artist_id

        super(PaintingsFolder, self).__init__(root=root, transform=transform, target_transform=target_transform,
                                              is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:

        all_artists = set(name for name in os.listdir(directory)
                          if os.path.isdir(os.path.join(directory, name)))

        classes = list(all_artists & set(self.artists_dict.keys()))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = deepcopy(self.artists_dict)

        del self.artists_dict
        return classes, class_to_idx


# use this class if you have a directory directly containing images
# and said images' classes are not relevant for the task
# rather than sub-directories representing the classes of the images 
class ClasslessImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = [os.path.join(root, file_name) for file_name in os.listdir(root) if os.path.isfile(os.path.join(root, file_name))]
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        y = 0
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.image_paths)
