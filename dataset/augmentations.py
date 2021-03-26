import albumentations as albu
import cv2
from config import new_augs

size = 1024
size_after_reshape = 320
ALBUMENTATIONS_VAL = albu.Compose([albu.Resize(size_after_reshape, size_after_reshape)])
if not new_augs:
    ALBUMENTATIONS_TRAIN = albu.Compose([
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.RandomRotate90(),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                            border_mode=cv2.BORDER_REFLECT),
        albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=.1),
            #albu.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
        albu.OneOf([
            albu.HueSaturationValue(10, 15, 10),
            albu.CLAHE(clip_limit=2),
            albu.RandomBrightnessContrast(),
            ], p=0.3),
        albu.Resize(size_after_reshape, size_after_reshape),
        ])
else:
    ALBUMENTATIONS_TRAIN = albu.Compose([
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.RandomRotate90(),
            albu.Cutout(num_holes=10,
                            max_h_size=int(.1 * size), max_w_size=int(.1 * size),
                            p=.25),
        albu.OneOf([
            albu.GaussNoise(0.002, p=.5),
            albu.IAAAffine(p=.5),
        ], p=.25),
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1),
            albu.MedianBlur(blur_limit=3, p=1)
        ], p=.25),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                              border_mode=cv2.BORDER_REFLECT),
        albu.HueSaturationValue(20, 30, 20),
        albu.RandomBrightnessContrast(),
        albu.OneOf([
            albu.OpticalDistortion(1.2, 0.5, p=.3),
            albu.GridDistortion(5, 0.2, p=.3),
            # albu.IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.RandomGamma(),
        ], p=0.5),
        albu.Resize(size_after_reshape, size_after_reshape),
    ])