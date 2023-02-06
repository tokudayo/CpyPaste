import argparse
import os
import cv2
import albumentations as A

from cpypaste.fix_aug import BgAwareSafeRotate, CoarseDropoutKeepBbox, Erode, ReplayCompose, ColorJitter
from cpypaste.paste import CopyPaste
from cpypaste.source import AutoAnnoImageFolder, ImageFolderBackgroundSource, ObjectSources
from cpypaste.utils import cv2_preview
from utils import gen_yolov5_data


def pre(img):
    # pad to square
    h, w = img.shape[:2]
    if h > w:
        img = cv2.copyMakeBorder(img, 0, 0, (h - w) // 2, (h - w) // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    else:
        img = cv2.copyMakeBorder(img, (w - h) // 2, (w - h) // 2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    img = cv2.resize(img, (256, 256))
    return img


obj_path = './assets/obj'
bg_path = '../capgen/assets/bg'

mapping = ['pawn', 'bishop', 'knight', 'rook', 'queen', 'king']
# weight = [0.5, 0.125, 0.125, 0.125, 0.0625, 0.0625]
weight = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_obj', type=int, default=1)
    parser.add_argument('--max_obj', type=int, default=34)
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--outdir', type=str, default='./test')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--min_scale', type=float, default=-0.6)
    parser.add_argument('--max_scale', type=float, default=0.0)
    args = parser.parse_args()
    config = dict(
        min_objects = args.min_obj,
        max_objects = args.max_obj,
    )

    obj_aug = ReplayCompose([
        # A.GaussNoise(p=1, per_channel=False),
        Erode(p=.2, kernel_size=3),
        A.RandomScale([args.min_scale, args.max_scale], p=1.),
        A.OneOf([
            # CoarseDropoutKeepBbox(p=0.5, max_holes=2, max_height=16, max_width=16, min_holes=1, min_height=4, min_width=4, fill_value=(0,0,0,255)),
            CoarseDropoutKeepBbox(p=0.5, max_holes=4, max_height=16, max_width=16, min_holes=1, min_height=4, min_width=4, fill_value=(0,0,0,0)),
        ], p=0.2),
        A.Perspective(scale=(0.05, 0.1), p=0.8,
                        keep_size=True, fit_output=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=(0, 0, 0, 0)),
        BgAwareSafeRotate(limit=90, p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.OneOf([
            A.MedianBlur(p=0.25, blur_limit=5),
            A.MotionBlur(p=0.25, blur_limit=5),
            A.Blur(p=0.25, blur_limit=5),
            A.GaussianBlur(p=0.25),
        ], p=0.2),
        A.OneOf([
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5),
            # A.ElasticTransform(p=1/3),
        ], p=0.1),
        ColorJitter(p=0.5, hue=0.1, saturation=0.7, brightness=0.7, contrast=0.7),
    ],
        bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0,),
        # keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        ignore_warnings=True
    )
    bg_aug = A.Compose([
        A.RandomResizedCrop(640, 640, p=1.),
    ])

    obj_sources = []
    for idx, piece_name in enumerate(mapping):
        obj_sources.append(
            AutoAnnoImageFolder(
                os.path.join(obj_path, piece_name),
                label_types=['mask', 'bboxes',],
                img_preprocess=pre,
                transform=obj_aug,
                png_auto_crop=True,
                cache_images=True,
                cache_labels=True,
                save_crop=False,
                bbox_class_idx=idx,
                load_cache=True,
                save_cache=True,
                cache_dir=f'./assets/{piece_name}_cache.npz'
            )
        )

    object_source = ObjectSources(obj_sources, sampling_weights=weight, )
    bg = ImageFolderBackgroundSource(bg_path, transform=bg_aug)
    cp = CopyPaste(object_source, bg, replay=False, replay_prob=1)

    gen_yolov5_data(cp, config, args.outdir, args.num, start_idx = args.start_idx)
    for source in obj_sources:
        source.write_cache()