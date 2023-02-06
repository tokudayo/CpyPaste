from typing import Any, Callable, Dict, Sequence, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from cpypaste.fix_aug import BgAwareSafeRotate, CoarseDropoutKeepBbox, ColorJitter, ReplayCompose, ReplayGuard
from cpypaste.source import AutoAnnoImageFolder, BackgroundSource, ImageFolderBackgroundSource, ObjectSource, ObjectSources

from cpypaste.utils import cv2_preview

# create a CP object
#     - generation configs:
#         - generate from one or several sources
#         - mixing object source and bg source + weight for each
#         - transformation for each
#         - how many objects of what type per bg?
#     - random config: all sources, weight, use the same tf for object and bg
# TODO: Overlap fix, class anno, meta, bbox and kp formats


class CopyPaste:
    def __init__(
        self,
        object_source: ObjectSource,
        background_source: BackgroundSource,
        config: Dict[str, Any] = None,
        transform_all: Callable = None,
        remove_invisible_keypoints: bool = True,
        replay: bool = False,
        replay_prob: float = 0.5,
    ):
        self._obj_src = object_source
        self._bg_src = background_source
        self.transform_all = transform_all
        self._config = config
        self._rm_invis = remove_invisible_keypoints
        self._replay_aug = replay
        self._replay_p = replay_prob

    def get_random_bg(self):
        return self._bg_src.get_random()
    
    def get_random_obj(self):
        return self._obj_src.get_random()

    def _calculate_coords(self, obj_dim: Tuple[int, int], bg_dim: Tuple[int, int], x: int, y: int) -> Sequence[int]:
        o_H, o_W = obj_dim
        b_H, b_W = bg_dim
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = np.clip(x + o_W, 0, b_W)
        y1 = np.clip(y + o_H, 0, b_H)
        x0_o = np.clip(-x, 0, o_W - 1)
        y0_o = np.clip(-y, 0, o_H - 1)
        x1_o = x0_o + x1 - x0
        y1_o = y0_o + y1 - y0
        return x0, y0, x1, y1, x0_o, y0_o, x1_o, y1_o

    def _update_mask(self, obj: Dict[str, Any], bg: Dict[str, Any], x: int, y: int, obj_num: int, inplace: bool = True) -> Dict[str, Any]:
        obj_mask = obj['mask']
        bg_mask = bg['mask'] if inplace else bg['mask'].copy()
        x0, y0, x1, y1, x0_o, y0_o, x1_o, y1_o = self._calculate_coords(obj_mask.shape[:2], bg_mask.shape[:2], x, y)
        idx = np.argwhere(obj_mask[y0_o:y1_o, x0_o:x1_o] > 0)
        bg_mask[y0:y1, x0:x1][idx[:, 0], idx[:, 1]] = obj_num
        return bg_mask

    def _update_keypoints(self, obj: Dict[str, Any], bg: Dict[str, Any], x: int, y: int, 
            inplace: bool = True, remove_masked: bool = True,) -> Dict[str, Any]:
        obj_kps = obj['keypoints']
        bg_kps = bg['keypoints'] if inplace else bg['keypoints'].copy()
        x0, y0, x1, y1, x0_o, y0_o, x1_o, y1_o = self._calculate_coords(obj['image'].shape[:2], bg['image'].shape[:2], x, y)
        if remove_masked:
            if 'mask' not in obj:
                raise ValueError('Mask is required for removing masked keypoints.')
            for idx, kp in enumerate(bg_kps):
                kp_x, kp_y = (int(c) for c in kp[:2])
                if x0 <= kp_x <= x1 and y0 <= kp_y <= y1 and obj['mask'][y0_o + kp_y - y0 - 1, x0_o + kp_x - x0 - 1] > 0:
                    del bg_kps[idx]
        for keypoint in obj_kps:
            keypoint = np.array(keypoint)
            keypoint[:2] += np.array([x, y])
            bg_kps.append(keypoint)
        return bg_kps

    def _update_bboxes(self, obj: Dict[str, Any], bg: Dict[str, Any], x: int, y: int, inplace: bool = True) -> Sequence[Sequence[float]]:
        bg_bboxes = bg['bboxes'] if inplace else bg['bboxes'].copy()
        for bbox in obj['bboxes']:
            bbox = np.array(bbox)
            bbox[:2] += np.array([x, y])
            bbox[2:4] += np.array([x, y])
            bg_bboxes.append(bbox)
        return bg_bboxes
            
    def _update_label(self, obj: Dict[str, Any], bg: Dict[str, Any], x: int, y: int, obj_num: int) -> None:
        label_types = set(obj.keys()) & {'bboxes', 'keypoints', 'mask'}
        if 'mask' in label_types and 'mask' not in bg:
            bg['mask'] = np.zeros_like(bg['image'][..., 0]).astype(np.uint8)
        if 'bboxes' in label_types and 'bboxes' not in bg:
            bg['bboxes'] = []
        if 'keypoints' in label_types and 'keypoints' not in bg:
            bg['keypoints'] = []
        if 'mask' in label_types:
            self._update_mask(obj, bg, x, y, obj_num, inplace=True)
        if 'bboxes' in label_types:
            self._update_bboxes(obj, bg, x, y, inplace=True)
        if 'keypoints' in label_types:
            self._update_keypoints(obj, bg, x, y, inplace=True)

    def paste_object(self, obj: Dict[str, Any], bg: Dict[str, Any], x: int, y: int, inplace: bool = False, fit_check: bool = False) -> Dict[str, Any]:
        if fit_check:
            if x + obj.shape[1] > bg.shape[1] or y + obj.shape[0] > bg.shape[0] \
                or x < 0 or y < 0:
                raise ValueError('Object does not fit in background.')
        if x >= bg.shape[1] or y >= bg.shape[0] or x + obj.shape[1] <= 0 or y + obj.shape[0] <= 0:
            return bg

        x0, y0, x1, y1, x0_o, y0_o, x1_o, y1_o = self._calculate_coords(obj.shape[:2], bg.shape[:2], x, y)
        if not inplace:
            bg = bg.copy()
        if obj.shape[2] < 4:
            bg[y0:y1, x0:x1] = obj[y0_o:y1_o, x0_o:x1_o, :3]
        else:
            alpha = obj[y0_o:y1_o, x0_o:x1_o, 3:4] / 255.0
            bg[y0:y1, x0:x1] = obj[y0_o:y1_o, x0_o:x1_o, :3] * alpha + bg[y0:y1, x0:x1] * (1 - alpha)
        return bg

    def _fix_bboxes_using_mask(self, gen: Dict[str, Any]):
        mask = gen['mask']
        bboxes = []
        for obj_num in np.unique(mask)[1:]:
            obj_mask = mask == obj_num
            y, x = np.where(obj_mask)
            bboxes.append([x.min(), y.min(), x.max(), y.max(), gen['bboxes'][obj_num - 1][-1]])
        gen['bboxes'] = bboxes
        return gen

    def generate(self, config: Dict[str, Any] = None, force_object_fit: bool = True, **kwargs) -> Dict[str, Any]:
        '''
        1 group of objects per image first
        '''
        # Get background
        bg = self.get_random_bg()
        bg = {'image': bg}

        # Get objects, random for now
        min_objects = config.get('min_objects', 1)
        max_objects = config.get('max_objects', 1)
        n_objects = np.random.randint(min_objects, max_objects + 1)
        with ReplayGuard(enable=self._replay_aug and np.random.rand() < self._replay_p):
            for i in range(n_objects):
                # Get a random object
                obj = self.get_random_obj()    
                # Random coordinates
                x = np.random.randint(0, bg['image'].shape[1] - obj['image'].shape[1]) \
                    if force_object_fit else np.random.randint(-obj['image'].shape[1] + 1, bg['image'].shape[1] - 1)
                y = np.random.randint(0, bg['image'].shape[0] - obj['image'].shape[0]) \
                    if force_object_fit else np.random.randint(-obj['image'].shape[0] + 1, bg['image'].shape[0] - 1)
                # Paste object + update label
                bg['image'] = self.paste_object(obj['image'], bg['image'], x, y, inplace=True)
                self._update_label(obj, bg, x, y, i + 1)
        # Fix bboxes using mask
        if 'bboxes' in bg and 'mask' in bg:
            self._fix_bboxes_using_mask(bg)
        return bg

config = dict(
    min_objects = 5,
    max_objects = 5,
    min_scale = 0.5,
    max_scale = 1,
)

if __name__ == '__main__':
    import albumentations as A

    def pre(img):
        # img = cv2.resize(img, (64, 64))
        return img
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    obj_aug = ReplayCompose([
        A.Resize(128, 128),
        # A.SafeRotate(limit=0, p=1, ),
        
        A.Perspective(scale=(0.05, 0.1), p=1,
                      keep_size=True, fit_output=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=(0, 0, 0, 0)),
        BgAwareSafeRotate(limit=90, p=1, border_mode=cv2.BORDER_CONSTANT, value=0), # NOTE: Shid rotate, fix later
        # ColorJitter(p=0.9),
        A.RandomScale([0.8, 0.0], p=1.),
        # A.GaussNoise(p=0.1),
        # CoarseDropoutKeepBbox(p=0.8, max_holes=24, max_height=64, max_width=64, min_holes=1, min_height=16, min_width=16, fill_value=0),
        
    ],
        bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0,),
        # keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        ignore_warnings=True
    )

    s1 = AutoAnnoImageFolder(
        './assets/obj/f1', label_types=['mask', 'bboxes',], img_preprocess=pre, transform=obj_aug,)
    s2 = AutoAnnoImageFolder(
        './assets/obj/f2', label_types=['mask', 'bboxes',], img_preprocess=pre, transform=obj_aug, bbox_class_idx=1)
    object_source = ObjectSources([s1, s2], sampling_weights=[0.9, 0.1])
    stuff = object_source.get_random()

    bg = ImageFolderBackgroundSource('./assets/bg')
    cp = CopyPaste(object_source, bg, replay=True)
    
    pasted = cp.generate(config, force_object_fit=True)
    cv2_preview(pasted, seperate_mask=True)