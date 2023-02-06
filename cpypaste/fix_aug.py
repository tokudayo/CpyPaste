from typing import Any, Dict
import cv2
import albumentations as A
import numpy as np
import random as py_random

def get_random_state() -> np.random.RandomState:
    return np.random.RandomState(py_random.randint(0, (1 << 32) - 1))

def uniform(low=0.0, high=1.0, size=None, random_state=None,) -> np.ndarray:
    if random_state is None:
        random_state = get_random_state()
    return random_state.uniform(low, high, size)

def normal(loc=0.0, scale=1.0, size=None, random_state=None,) -> np.ndarray:
    if random_state is None:
        random_state = get_random_state()
    return random_state.normal(loc, scale, size)

class ReplayCompose(A.ReplayCompose):
    _replay = False
    _played_once = False
    def __init__(self, *args, ignore_warnings: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if ignore_warnings:
            import warnings
            warnings.filterwarnings('ignore', message='.*could work incorrectly in ReplayMode.*')

    def _set_replay(self, flag: bool, params: dict = None) -> None:
        for tf in self.transforms:
            tf.replay_mode = flag
            tf.applied_in_replay = flag
            if flag and params is not None:
                for tr in params:
                    if tr['__class_fullname__'].split('.')[-1] == tf.__class__.__name__:
                        tf.params = tr['params']
                        break

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        if ReplayCompose._replay:
            if not ReplayCompose._played_once:
                ReplayCompose._played_once = True
                self._set_replay(False)
                results = super().__call__(*args, **kwargs)
                self._set_replay(True, results['replay']['transforms'])
                results.pop('replay')
                return results
            else:
                return super().__call__(*args, **kwargs)
        else:
            if not ReplayCompose._played_once:
                self._set_replay(False)
                ReplayCompose._played_once = True
            return super().__call__(*args, **kwargs)

class ReplayGuard:
    def __init__(self, enable: bool = True):
        self._cur_state = ReplayCompose._replay
        self._state = enable

    def __enter__(self):
        ReplayCompose._replay = self._state
        ReplayCompose._played_once = False
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        ReplayCompose._replay = self._cur_state
        ReplayCompose._played_once = False

class Erode(A.ImageOnlyTransform):
    def __init__(self, kernel_size: int = 3, p: float = 1.0, always_apply: bool = False,):
        super().__init__(always_apply, p)
        self.kernel_size = kernel_size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if isinstance(self.kernel_size, list):
            kernel_size = np.random.choice(self.kernel_size)
        else:
            kernel_size = self.kernel_size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(img, kernel, iterations=1)

    def get_transform_init_args_names(self) -> tuple:
        return ('kernel_size',)

class ColorJitter(A.ColorJitter):
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.shape[2] == 4:
            alpha = img[..., 3]
            img = img[..., :3]
        else:
            alpha = None
        img = super().apply(img, **params)
        if alpha is not None:
            img = np.dstack((img, alpha))

        return img

class BgAwareSafeRotate(A.SafeRotate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        kwargs = super().__call__(*args, **kwargs)
        H, W, _ = kwargs['image'].shape
        if kwargs['image'].shape[2] == 4 and 'bboxes' in kwargs:
            mask = kwargs['image'][..., 3] > 0
            min_y, min_x = np.min(np.argwhere(mask), axis=0).astype(float)
            max_y, max_x = np.max(np.argwhere(mask), axis=0).astype(float)
            extra = kwargs['bboxes'][0][4:]
            kwargs['bboxes'][0] = (min_x/W, min_y/H, max_x/W, max_y/H) + extra
        return kwargs
        
class CoarseDropoutKeepBbox(A.CoarseDropout):
    def apply_to_bbox(self, bbox: A.BoxType, **params) -> A.BoxType:
        return bbox

class Perspective(A.Perspective):
    def _sign (self, p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);

    def _in_triangle(self, pt, v1, v2, v3):
        d1 = self._sign(pt, v1, v2)
        d2 = self._sign(pt, v2, v3)
        d3 = self._sign(pt, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)
    
    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]
        _in_tr = self._in_triangle
        while True:
            scale = uniform(*self.scale)
            points = normal(0, scale, [4, 2])
            points = np.mod(np.abs(points), 1)

            # top left -- no changes needed, just use jitter
            # top right
            points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
            # bottom right
            points[2] = 1.0 - points[2]  # w = 1.0 - jitt
            # bottom left
            points[3, 1] = 1.0 - points[3, 1]  # h = 1.0 - jitter

            points[:, 0] *= w
            points[:, 1] *= h
            pt1, pt2, pt3, pt4 = points
            if not (_in_tr(pt1, pt2, pt3, pt4) or _in_tr(pt2, pt1, pt3, pt4) or \
                        _in_tr(pt3, pt1, pt2, pt4) or _in_tr(pt4, pt1, pt2, pt3)):
                    break

        # Obtain a consistent order of the points and unpack them individually.
        # Warning: don't just do (tl, tr, br, bl) = _order_points(...)
        # here, because the reordered points is used further below.
        points = self._order_points(points)
        tl, tr, br, bl = points

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        min_width = None
        max_width = None
        while min_width is None or min_width < 2:
            width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            max_width = int(max(width_top, width_bottom))
            min_width = int(min(width_top, width_bottom))
            if min_width < 2:
                step_size = (2 - min_width) / 2
                tl[0] -= step_size
                tr[0] += step_size
                bl[0] -= step_size
                br[0] += step_size

        # compute the height of the new image, which will be the maximum distance between the top-right
        # and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        min_height = None
        max_height = None
        while min_height is None or min_height < 2:
            height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = int(max(height_right, height_left))
            min_height = int(min(height_right, height_left))
            if min_height < 2:
                step_size = (2 - min_height) / 2
                tl[1] -= step_size
                tr[1] -= step_size
                bl[1] += step_size
                br[1] += step_size

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left order
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        dst = np.array([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]], dtype=np.float32)

        # compute the perspective transform matrix and then apply it
        m = cv2.getPerspectiveTransform(dst, points)
        print(f"points: {points}, dst: {dst}")

        if self.fit_output:
            m, max_width, max_height = self._expand_transform(m, (h, w))

        return {"matrix": m, "max_height": max_height, "max_width": max_width, "interpolation": self.interpolation}
