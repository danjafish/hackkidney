import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from config import *
from utils.support_func import *
from nn.trainer import *
import segmentation_models_pytorch as smp
import gc
from dataset.dataloader import KidneySampler, KidneyLoader
import tifffile as tiff
from torch.optim import AdamW, Adam
from torch.nn import BCEWithLogitsLoss
import os

if __name__ == '__main__':
    model_path = ''
    overlap = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    print('Start data preparation')
    seed_everything(2021)
    data_path = '../data/'
    data = pd.read_csv(data_path + 'train.csv')
    model = smp.Unet("efficientnet-b4", encoder_weights="imagenet", in_channels=3, classes=1,
                     decoder_use_batchnorm=False).cuda()

    sample_sub = pd.read_csv(data_path + 'sample_submission.csv')
    test_paths = sample_sub.id.values
    print('Start test')

    X_test_images = []
    img_dims_test = []
    for name in test_paths:
        img = tiff.imread(data_path + f"test/{name}.tiff")
        if img.shape[0] == 3:
            img = np.moveaxis(img, 0, 2)
        X_test_images.append(img)
        img_dims_test.append(img.shape)
        print(img.shape)
    del img
    gc.collect()

    test_dataset = ValLoader(X_test_images, img_dims_test, size, overlap=True, step_size=step_size)
    testloader = DataLoader(test_dataset, batch_size=bs * 2, shuffle=False, num_workers=16)
    model.load_state_dict(load(f'../{model_path}/last_best_model.h5'))
    test_masks, test_keys = predict_test(model, size, testloader, True)
    all_enc = []
    for n in range(len(sample_sub)):
        img_n_keys = [(i, k) for i, k in enumerate(test_keys) if k[0] == n]
        mask = mask_from_keys_and_preds_test(img_n_keys, test_masks, n, img_dims_test,
                                             size, step_size=step_size, overlap=True)
        t = 0.4
        mask[mask < t] = 0
        mask[mask >= t] = 1
        enc = mask2enc(mask)
        all_enc.append(enc[0])
    sample_sub.predicted = all_enc
    sample_sub.to_csv(f'../{model_path}/best_{model_name}_{t}_{overlap}.csv', index=False)
