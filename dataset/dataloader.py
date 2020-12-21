from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from dataset.augmentations import ALBUMENTATIONS_TRAIN, ALBUMENTATIONS_VAL
import numpy as np

from torch.utils.data import Dataset, DataLoader


class ValLoader(Dataset):
    def __init__(self, images, piece_dim=512):
        self.piece_dim = piece_dim
        self.n_images = len(images)
        self.images = images
        image_dims = [(31295, 40429), (14844, 31262), (38160, 42360), (26840, 49780), (36800, 43780)]
        # assert len(images) == len(image_dims)
        self.ids = [(image_id, x, y)
                    for image_id in range(self.n_images)
                    for x in range(image_dims[image_id][0] // self.piece_dim)
                    for y in range(image_dims[image_id][1] // self.piece_dim)
                    ]

    def __getitem__(self, idx):
        image_id, x, y = self.ids[idx]

        img = self.images[image_id]
        piece = img[x * self.piece_dim: (x + 1) * self.piece_dim,
                y * self.piece_dim: (y + 1) * self.piece_dim]
        aug = ALBUMENTATIONS_VAL(image=piece)
        piece = aug['image']

        return piece, self.ids[idx]

    def __len__(self):
        return len(self.ids)


class KidneySampler(Sampler):
    def __init__(self, positive_idxs, negative_idxs, demand_non_empty_proba=0.5):
        assert demand_non_empty_proba > 0, 'frequensy of non-empty images must be greater then zero'
        self.positive_proba = demand_non_empty_proba

        self.positive_idxs = positive_idxs
        self.negative_idxs = negative_idxs

        self.n_positive = len(positive_idxs)
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)
        print(self.n_positive, self.n_negative)

    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative


class KidneyLoader(Dataset):
    def __init__(self, images, masks, image_dims, val=False, piece_dim=512, step_size=0, val_index=0):
        self.piece_dim = piece_dim
        self.n_images = len(images)
        self.images = images
        self.masks = masks
        self.step_size = step_size
        self.val = val
        #         image_dims = [(31278, 25794), (18484, 13013), (34940, 49548), (25784, 34937),
        #                      (16180, 27020), (38160, 39000), (30440, 22240), (26780, 32220)]

        # assert len(images) == len(image_dims)
        if (self.val or self.step_size == 0):
            self.ids = [(image_id, x, y)
                        for image_id in range(self.n_images)
                        for x in range(image_dims[image_id][0] // self.piece_dim)
                        for y in range(image_dims[image_id][1] // self.piece_dim)
                        ]
        else:
            self.ids = [(image_id, x, y)
                        for image_id in range(self.n_images)
                        for x in range(0, (image_dims[image_id][0] // self.step_size) - 1)
                        for y in range((image_dims[image_id][1] // self.step_size) - 1)
                        ]

        self.val_idx = val_index
        if self.val:
            self.ids = [x for x in self.ids if x[0] == self.val_idx]
        else:
            self.ids = [x for x in self.ids if x[0] != self.val_idx]

    def __getitem__(self, idx):
        image_id, x, y = self.ids[idx]

        img = self.images[image_id]
        mask = self.masks[image_id]
        if (self.val or self.step_size == 0):
            piece = img[x * self.piece_dim: (x + 1) * self.piece_dim,
                    y * self.piece_dim: (y + 1) * self.piece_dim]
            mask = mask[x * self.piece_dim: (x + 1) * self.piece_dim,
                   y * self.piece_dim: (y + 1) * self.piece_dim]
        else:
            piece = img[x * self.step_size: x * self.step_size + self.piece_dim,
                    y * self.step_size: y * self.step_size + self.piece_dim]
            mask = mask[x * self.step_size: x * self.step_size + self.piece_dim,
                   y * self.step_size: y * self.step_size + self.piece_dim]
        if not self.val:
            aug = ALBUMENTATIONS_TRAIN(image=piece, mask=mask)
            piece = aug['image']
            mask = aug['mask']
        else:
            aug = ALBUMENTATIONS_VAL(image=piece, mask=mask)
            piece = aug['image']
            mask = aug['mask']

        return piece, np.expand_dims(mask, axis=0), self.ids[idx]

    def __len__(self):
        return len(self.ids)