from functools import partial
import os
from typing import Any, Callable, Dict, List, Literal, Sequence
import warnings
import cv2
import numpy as np


class ObjectSource:
    def __init__(self, seed: int = None, img_preprocess: Callable = None, transform: Callable = None):
        self._prng = np.random.RandomState(seed)
        self._img_preprocess = img_preprocess
        self._transform = transform

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Should return a dict in form of
        {
            'image': np.ndarray (H, W, C),
            'keypoints': np.ndarray (N, (x, y, ...)),
            'bbox': np.ndarray (N, (x1, y1, x2, y2, ...)),
            'mask': (N, H, W),
            'extra': Any,
        }
        """
        raise NotImplementedError

    def get_random(self) -> Dict[str, Any]:
        return self[self._prng.randint(0, len(self))]

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        if self._img_preprocess is not None:
            return self._img_preprocess(img)
        return img

    def transform(self, im_label: Dict[str, Any]) -> Dict[str, Any]:
        if self._transform is not None:
            return self._transform(**im_label)
        return im_label


class AutoAnnoImageFolder(ObjectSource):
    __slots__ = ['_prng', '_path', '_img_list', '_label_types', '_cache_images', '_cache_labels',
                 '_png_auto_crop', '_save_crop', '_images', '_labels', '_need_cropping', '_class_idx']

    def __init__(
        self,
        path: str,
        bbox_class_idx: int = 0,
        label_types: Sequence[Literal['bboxes',
                                   'keypoints', 'mask']] = ['bboxes'],
        cache_images: bool = True,
        cache_labels: bool = True,
        load_cache: bool = False,
        save_cache: bool = False,
        cache_dir: str = None,
        png_auto_crop: bool = True,
        save_crop: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._path = path
        self._img_list = os.listdir(path)
        self._label_types = label_types
        self._cache_images = cache_images
        self._cache_labels = cache_labels
        self._save_cache = save_cache
        self._cache_dir = cache_dir

        if self._cache_images:
            self._images = [None] * len(self._img_list)
        if self._cache_labels:
            self._labels = [None] * len(self._img_list)
        if load_cache:
            try:
                cache = np.load(cache_dir, allow_pickle=True)
                self._images = cache['images']
                self._labels = cache['labels']
            except FileNotFoundError:
                warnings.warn(f'Cache file {cache_dir} not found')
                pass

        self._png_auto_crop = png_auto_crop
        if png_auto_crop:
            self._need_cropping = np.ones(len(self._img_list), dtype=bool)
        self._save_crop = save_crop

        self._class_idx = bbox_class_idx

    def __len__(self) -> int:
        return len(self._img_list)

    def _get_image(self, idx: int) -> np.ndarray:
        if self._cache_images and self._images[idx] is not None:
            return self._images[idx]

        img = cv2.imread(
            f'{self._path}/{self._img_list[idx]}', cv2.IMREAD_UNCHANGED) # BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        

        if img.shape[2] == 4 and self._png_auto_crop:
            if not self._need_cropping[idx]:
                return img
            mask = img[..., 3] > 0
            min_y, min_x = np.min(np.argwhere(mask), axis=0)
            max_y, max_x = np.max(np.argwhere(mask), axis=0)
            img = img[min_y:max_y + 1, min_x:max_x + 1]


            if (min_x == 0 and min_y == 0 and max_x == img.shape[1] - 1 and max_y == img.shape[0] - 1) \
                    or self._save_crop:
               self._need_cropping[idx] = False
        
        img = self.preprocess_img(img)

        if self._save_crop:
            cv2.imwrite(f'{self._path}/{self._img_list[idx]}', img)

        if self._cache_images:
            self._images[idx] = img

        return img

    def _get_label(self, img: np.ndarray, idx: int) -> Dict[str, np.ndarray]:
        if self._cache_labels and self._labels[idx] is not None:
            return self._labels[idx]

        labels = {}
        if 'mask' in self._label_types:
            labels['mask'] = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) \
                if img.shape[2] < 4 else (img[..., 3] > 0).astype(np.uint8)
        if 'bboxes' in self._label_types:
            if 'mask' in labels:
                mask = labels['mask']
                min_y, min_x = np.min(np.argwhere(mask), axis=0)
                max_y, max_x = np.max(np.argwhere(mask), axis=0)
                labels['bboxes'] = np.array(
                    [[min_x, min_y, max_x, max_y, self._class_idx]])
            else:
                labels['bboxes'] = np.array(
                    [[0, 0, img.shape[1] - 1, img.shape[0] - 1, self._class_idx]])
        if 'keypoints' in self._label_types:
            labels['keypoints'] = np.array([
                [0, 0], [img.shape[1] - 1, 0], 
                [img.shape[1] -1, img.shape[0] - 1], [0, img.shape[0] - 1]
            ])

        if self._cache_labels:
            self._labels[idx] = labels

        return labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = self._get_image(idx)
        labels = self._get_label(img, idx)
        return self.transform({'image': img, **labels})
    
    def write_cache(self):
        if self._save_cache:
            np.savez(self._cache_dir, images=self._images, labels=self._labels)


class ObjectSources(ObjectSource):
    @staticmethod
    def from_folder(path: str) -> 'ObjectSources':
        subfolder_names = [name for name in os.listdir(
            path) if os.path.isdir(f'{path}/{name}')]
        return ObjectSources(
            [AutoAnnoImageFolder(f'{path}/{subfolder_name}')
             for subfolder_name in subfolder_names]
        )

    def __init__(self, sources: List[ObjectSource], sampling_weights: List[float] = None, **kwargs):
        super().__init__(**kwargs)
        self._sources = sources
        self._weights = np.full(len(sources), 1 / len(sources)) \
            if sampling_weights is None else sampling_weights
        assert len(self._weights) == len(
            self._sources), 'Sampling weights array must match number of sources.'
        assert np.isclose(np.sum(self._weights),
                          1), 'Sampling weights must sum to 1.'

    def __len__(self) -> int:
        return sum([len(source) for source in self._sources])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        source_idx = 0
        while idx >= len(self._sources[source_idx]):
            idx -= len(self._sources[source_idx])
            source_idx += 1
        return self._sources[source_idx][idx]

    def get_from_source(self, source_idx: int) -> Dict[str, Any]:
        return self._sources[source_idx].get_random()

    def _random_source(self) -> AutoAnnoImageFolder:
        return self._sources[self._prng.choice(len(self._sources), p=self._weights)]

    def get_random(self) -> Dict[str, Any]:
        return self._random_source().get_random()

    @property
    def num_sources(self) -> int:
        return len(self._sources)


class BackgroundSource:
    def __init__(self, seed: int = None, transform: Callable = None) -> None:
        self._prng = np.random.RandomState(seed)
        self._transform = transform

    def get_random(self) -> np.ndarray:
        return self[self._prng.randint(len(self))]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self) -> np.ndarray:
        raise NotImplementedError

    def transform(self, img: np.ndarray) -> np.ndarray:
        if self._transform is not None:
            return self._transform(image=img)['image']
        return img

class ImageFolderBackgroundSource(BackgroundSource):
    def __init__(self, path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path = path
        self._img_list = os.listdir(path)

    def __len__(self) -> int:
        return len(self._img_list)

    def __getitem__(self, idx: int) -> np.ndarray:
        im = cv2.imread(f'{self._path}/{self._img_list[idx]}')
        return self.transform(
            cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        )
