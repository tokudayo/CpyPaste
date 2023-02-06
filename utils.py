import os
from typing import Any, Dict
import warnings
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

from cpypaste.paste import CopyPaste



def pascal_voc_to_yolo(x_min, y_min, x_max, y_max, im_width, im_height):
    x = (x_min + x_max) / 2 / im_width
    y = (y_min + y_max) / 2 / im_height
    w = (x_max - x_min) / im_width
    h = (y_max - y_min) / im_height
    return x, y, w, h

def preview_yolo_bboxes(img, bboxes):
    for bbox in bboxes:
        x, y, w, h = bbox
        x1, y1 = int((x - w / 2) * img.shape[1]), int((y - h / 2) * img.shape[0])
        x2, y2 = int((x + w / 2) * img.shape[1]), int((y + h / 2) * img.shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

class IgnoreAllError:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, _, __, ___):
        pass

def gen_yolov5_data(cp_instance: CopyPaste, config: Dict[str, Any], outdir: str, size: int = 1000, start_idx: int = 0):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(os.path.join(outdir, 'images')):
        os.makedirs(os.path.join(outdir, 'images'))
    if not os.path.exists(os.path.join(outdir, 'labels')):
        os.makedirs(os.path.join(outdir, 'labels'))
    for i in tqdm(range(size)):
        out = cp_instance.generate(config)
        img = out['image']
        bboxes = out['bboxes']
        H, W = img.shape[:2]
        cv2.imwrite(os.path.join(outdir, 'images', f'{i + start_idx}.jpg'), img[..., ::-1])
        with open(os.path.join(outdir, 'labels', f'{i + start_idx}.txt'), 'w') as f:
            for box in bboxes:
                x_min, y_min, x_max, y_max, label = box
                label = int(label)
                x, y, w, h = pascal_voc_to_yolo(x_min, y_min, x_max, y_max, W, H)
                f.write(f"{label} {x} {y} {w} {h}\n")


import random
from multiprocessing import Pool

def iou_matrix(a, b, norm=True):
    tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    pad = not norm and 1 or 0

    area_i = np.prod(br_i - tl_i + pad, axis=2) * (tl_i < br_i).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2] + pad, axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2] + pad, axis=1)
    area_o = (area_a[:, np.newaxis] + area_b - area_i)
    return area_i / (area_o + 1e-10)

def _process_chunk(det_dir, out_dir, chunk, crop_limit=0.2, class_map=None, max_iou=0.4):
    if isinstance(crop_limit, (list, tuple)):
        crop_in, exp_out = crop_limit
    else:
        crop_in = exp_out = crop_limit
    for name in chunk:
        im = cv2.imread(os.path.join(det_dir, 'images', name))
        H, W = im.shape[:2]
        with open(os.path.join(det_dir, 'labels', name.replace('jpg', 'txt')), 'r') as f:
            lines = f.readlines()
        all_boxes = []
        labels = []
        for idx, line in enumerate(lines):
            line = line.strip().split()
            label = int(line[0])
            x, y, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            # Crop around the bbox
            x1, y1 = int((x - w / 2) * W), int((y - h / 2) * H)
            x2, y2 = int((x + w / 2) * W), int((y + h / 2) * H)
            all_boxes.append([x1, y1, x2, y2])
            labels.append(label)
        
        all_boxes = np.array(all_boxes)
        iou = iou_matrix(all_boxes, all_boxes)
        np.fill_diagonal(iou, 0)
        miou = np.max(iou, axis=1)
        selected_boxes = np.argwhere(miou < max_iou)
        if len(selected_boxes) == 0:
            continue
        selected_boxes = selected_boxes.flatten()
        all_boxes = all_boxes[selected_boxes]
        labels = np.array(labels)[selected_boxes]
        
        # IOU calculation
        for idx, box in enumerate(all_boxes):
            label = labels[idx]
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            # Random crop
            x1_offset = int(np.random.uniform(-exp_out, crop_in) * box_width)
            y1_offset = int(np.random.uniform(-exp_out, crop_in) * box_height)
            x2_offset = int(np.random.uniform(-crop_in, exp_out) * box_width)
            y2_offset = int(np.random.uniform(-crop_in, exp_out) * box_height)
            x1 += x1_offset
            y1 += y1_offset
            x2 += x2_offset
            y2 += y2_offset
            x1 = np.clip(x1, 0, W)
            y1 = np.clip(y1, 0, H)
            x2 = np.clip(x2, 0, W)
            y2 = np.clip(y2, 0, H)
            crop = im[y1:y2, x1:x2]
            if crop.shape[0] <= 10 or crop.shape[1] <= 10:
                continue
            cv2.imwrite(os.path.join(out_dir, str(class_map[label]), f'{name[:-4]}_{idx}.jpg'), crop)


def det_to_cls_dataset(det_dir, out_dir, no_classes, crop_limit=0.2, class_map=None, max_iou=0.4, size=20000, n_threads=8,):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    class_map = class_map or {i: str(i) for i in range(no_classes)}
    for folder_name in class_map.values():
        if not os.path.exists(os.path.join(out_dir, folder_name)):
            os.makedirs(os.path.join(out_dir, folder_name))

    im_name = os.listdir(os.path.join(det_dir, 'images'))
    print(len(im_name))
    random.shuffle(im_name)
    im_name = im_name[:size]
    with Pool(n_threads) as p:
        p.starmap(_process_chunk, [(det_dir, out_dir, im_name[i::n_threads], crop_limit, class_map, max_iou) for i in range(n_threads)])

    