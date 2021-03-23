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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    print('Start data preparation')
    seed_everything(2021)
    os.system(f"mkdir ../{str(model_name)}")
    #data_path = '/home/data/Kidney/data/'
    data_path = '../data/'
    data = pd.read_csv(data_path + 'train.csv')
    X_images = []
    Masks = []
    image_dims = []
    for i in range(len(data)):
        img = tiff.imread(data_path + f"train/{data.id[i]}.tiff")
        if img.shape[0] == 1:
            img = img[0][0]
        if img.shape[0] == 3:
            img = np.moveaxis(img, 0, 2)
        X_images.append(img)
        shape = (img.shape[1], img.shape[0])
        mask = enc2mask(data.loc[data.id == data.id[i], "encoding"].values[0], shape)
        Masks.append(mask)
        image_dims.append(img.shape)
        print(img.shape)

    del img, mask, shape
    gc.collect()
    positive_idxs, negative_idxs = get_indexes(val_index, X_images, Masks, image_dims, size, step_size)
    kid_sampler = KidneySampler(positive_idxs, negative_idxs, not_empty_ratio)
    train_dataset = KidneyLoader(X_images, Masks, image_dims, False, size, step_size=step_size, val_index=val_index)
    val_dataset = KidneyLoader(X_images, Masks, image_dims, True, size, val_index=val_index)
    if not use_sampler:
        print("Use full dataset")
        trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=16)
    else:
        print("Use sample dataset")
        trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=16, sampler=kid_sampler)
    valloader = DataLoader(val_dataset, batch_size=bs * 2, shuffle=False, num_workers=16)
    x, y, key = train_dataset[10]
    print(x.shape, y.shape)
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1,
                     decoder_use_batchnorm=False).cuda()
    if loss_name == 'comboloss':
        print("Use combo loss")
        loss = ComboLoss(weights=weights)
    else:
        print("Use BCE Loss")
        loss = BCEWithLogitsLoss()
    optim = AdamW(model.parameters(), lr=max_lr)
    metric = smp.utils.losses.DiceLoss()
    SchedulerClass_cos = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_params_cos = dict(
        T_max=epochs, eta_min=min_lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **scheduler_params_cos)
    # ---- Start train loop here ----- #
    print('Start train')
    min_val_loss = 100
    max_val_dice = -100
    for epoch in range(epochs + epochs_minlr):
        with open(f"../{model_name}/{model_name}.log", 'a+') as logger:
            logger.write(f"Epoch # {epoch}, lr = {optim.param_groups[0]['lr']}\n")
        empty_cache()
        dice = 0
        print(f"epoch number {epoch}, lr = {optim.param_groups[0]['lr']}")
        if (use_sampler & use_adaptive_sampler):
            if epoch < 10:
                kid_sampler = KidneySampler(positive_idxs, negative_idxs, 0.2)
                trainloader = DataLoader(train_dataset, batch_size=bs,
                                         shuffle=False, num_workers=16, sampler=kid_sampler)
            else:

                kid_sampler = KidneySampler(positive_idxs, negative_idxs, 0.5)
                trainloader = DataLoader(train_dataset, batch_size=bs,
                                         shuffle=False, num_workers=16, sampler=kid_sampler)

        model, optim, pred_keys, final_masks, \
        train_loss, train_dice = train_one_epoch(model, optim, trainloader, size, loss)
        print(f"train loss = {train_loss}")

        model, optim, val_keys, val_masks, val_loss = val_one_epoch(model, optim, valloader, size, loss)
        print(f"val loss = {val_loss}")

        if epoch < epochs:
            scheduler.step()
        val_dice = calc_average_dice(Masks, val_keys, val_masks, val_index, image_dims, size)
        if val_dice > max_val_dice:
            max_val_dice = val_dice
            save(model.state_dict(), f"../{model_name}/{model_name}_{epoch}.h5")
            save(model.state_dict(), f"../{model_name}/last_best_model.h5")

        print("Dice on train micro ", train_dice)
        print(f"Dice on val (average) = {val_dice}")
        with open(f"../{model_name}/{model_name}.log", 'a+') as logger:
            logger.write(f'train loss = {train_loss}, train dice (micro) = {train_dice}\n')
            logger.write(f'validation loss = {val_loss}, val dice = {val_dice}\n')
            logger.write('\n')
        print("=====================")
    model.load_state_dict(load(f"../{model_name}/last_best_model.h5"))
    val_keys, val_masks = predict_data(model, valloader, size, True)
    val_dice = calc_average_dice(Masks, val_keys, val_masks, val_index, image_dims, size)
    print(f"Dice on val (average) with TTA = {val_dice}")

    sample_sub = pd.read_csv(data_path + 'sample_submission.csv')
    test_paths = sample_sub.id.values
    print('Start test')
    del X_images
    gc.collect()
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

    test_dataset = ValLoader(X_test_images, img_dims_test, size)
    testloader = DataLoader(test_dataset, batch_size=bs * 2, shuffle=False, num_workers=16)
    model.load_state_dict(load(f'../{model_name}/last_best_model.h5'))
    test_masks, test_keys = predict_test(model, size, testloader, True)
    all_enc = []
    for n in range(len(sample_sub)):
        img_n_keys = [(i, k) for i, k in enumerate(test_keys) if k[0] == n]
        mask = mask_from_keys_and_preds_test(img_n_keys, test_masks, n, img_dims_test, size)
        t = 0.5
        mask[mask < t] = 0
        mask[mask >= t] = 1
        enc = mask2enc(mask)
        all_enc.append(enc[0])
    sample_sub.predicted = all_enc
    sample_sub.to_csv(f'../{model_name}/{model_name}.csv', index=False)
