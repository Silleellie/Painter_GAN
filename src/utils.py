import os
from copy import deepcopy
from typing import Tuple, List, Dict, Callable, Optional

from torchvision.datasets import ImageFolder


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

        classes = sorted(name for name in os.listdir(directory)
                         if os.path.isdir(os.path.join(directory, name)) and name in set(self.artists_dict.keys()))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = deepcopy(self.artists_dict)

        del self.artists_dict
        return classes, class_to_idx
