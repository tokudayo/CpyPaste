from typing import Any, Dict
import cv2
import numpy as np


color_code = [
    (128, 0, 0),
    (170, 110, 40),
    (128, 128, 0),
    (0, 128, 128),
    (0, 0, 128),
    (230, 25, 75),
    (245, 130, 48),
    (255, 255, 25),
    (210, 245, 60),
    (60, 180, 75),
    (70, 240, 240),
    (0, 130, 200),
    (145, 30, 180),
    (240, 50, 230),
    (128, 128, 128),
    (250, 190, 212),
    (255, 215, 180),
    (255, 250, 200),
    (170, 255, 195),
    (220, 190, 255),
    (255, 255, 255),
]

def cv2_preview(im_label: Dict[str, Any], wait_time: int = 0, seperate_mask: bool = True) -> None:
    img = im_label['image'].copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    if 'bboxes' in im_label:
        for idx, bbox in enumerate(im_label['bboxes']):
            color = color_code[idx % len(color_code)]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(img, str(bbox[4]), (int(bbox[0]), int(bbox[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if 'keypoints' in im_label:
        for kp in im_label['keypoints']:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), 2)

    if 'mask' in im_label:
        mask = im_label['mask']

        max_num = np.max(mask)
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for idx in range(1, max_num + 1):
            rgb_mask[mask == idx] = color_code[(idx - 1) % len(color_code)]
        bgr_mask = rgb_mask[..., ::-1]

    if 'mask' in im_label:
        if seperate_mask:
            cv2.imshow('mask', bgr_mask)
        else:
            img = cv2.addWeighted(img, 0.8, bgr_mask, 0.2, 0)

    cv2.imshow('preview', img[..., ::-1])
    cv2.waitKey(wait_time)
    cv2.destroyWindow('preview')
    if 'mask' in im_label and seperate_mask:
        cv2.destroyWindow('mask')


def cv2_show_alpha(img: np.ndarray, wait_time: int = 0) -> None:
    cv2.imshow('preview', img[..., 3])
    cv2.waitKey(wait_time)
    cv2.destroyWindow('preview')
