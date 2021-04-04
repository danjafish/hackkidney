import albumentations as albu
import cv2
import numpy as np


class CutMix:
    def __init__(self, p=1.0, max_h_size=80, max_w_size=80, full_mask=True):
        self.max_h = max_h_size
        self.max_w = max_w_size
        self.p = p
        self.full_mask = full_mask

    def transform(self, target_piece, target_mask, source_piece, source_mask):
        """
        Use subframes of source image to augment target image.
        """
        if np.random.rand() < self.p:
            # cut relevant piece from source
            source_submask, source_subpiece = self.get_subframe_with_mask(source_mask, source_piece)
            # get random position of target
            y_min, y_max, x_min, x_max = self.get_subframe_idx(source_shape=source_submask.shape,
                                                               target_shape=target_mask.shape)

            # create mixed image
            if self.full_mask:
                target_piece[y_min: y_max, x_min: x_max] = source_subpiece
                target_mask[y_min: y_max, x_min: x_max] = source_submask
            else:
                target_mask[y_min: y_max, x_min: x_max][source_submask != 0] = source_submask[source_submask != 0]
                target_piece[y_min: y_max, x_min: x_max][source_submask != 0] = source_subpiece[source_submask != 0]
            #

        return target_piece, target_mask

    def get_subframe_idx(self, source_shape, target_shape):
        y = np.random.randint(0, target_shape[0] - source_shape[0])
        x = np.random.randint(0, target_shape[1] - source_shape[1])
        return (
            y, y + source_shape[0],
            x, x + source_shape[1]
        )

    def get_subframe_with_mask(self, mask, piece):
        x_center = np.argmax(mask.sum(0))
        y_center = np.argmax(mask.sum(1))

        w_size = min(self.max_w // 2, x_center, mask.shape[0] - x_center)
        h_size = min(self.max_h // 2, y_center, mask.shape[1] - y_center)

        idx = (y_center - h_size, y_center + h_size, x_center - w_size, x_center + w_size)
        return self.get_slice(mask, *idx), self.get_slice(piece, *idx)

    def get_slice(self, x, y_min, y_max, x_min, x_max):
        return x[y_min: y_max, x_min: x_max]


def get_augs(new_augs, size, size_after_reshape):
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
    return ALBUMENTATIONS_TRAIN, ALBUMENTATIONS_VAL