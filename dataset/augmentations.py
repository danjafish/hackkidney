import albumentations as albu
import cv2
from config import size_after_reshape
ALBUMENTATIONS_VAL = albu.Compose([albu.Resize(size_after_reshape, size_after_reshape)])

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
        albu.HueSaturationValue(10,15,10),
        albu.CLAHE(clip_limit=2),
        albu.RandomBrightnessContrast(),
        ], p=0.3),
    albu.Resize(size_after_reshape, size_after_reshape),
    ])