import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils.support_func import *
from nn.trainer import *
import segmentation_models_pytorch as smp
import gc
from dataset.dataloader import KidneySampler, KidneyLoader
import tifffile as tiff
from torch.optim import AdamW, Adam
from torch.nn import BCEWithLogitsLoss
import os


def train_all(config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_number"])
    print('Start data preparation')
    seed_everything(2020)
    os.system(f"mkdir ../{model_name}")
    data_path = '/home/data/Kidney/data/'
    data = pd.read_csv(data_path + 'train.csv')
    X_images = []
    Masks = []
    image_dims = []
    for i in range(len(data)):
        img = tiff.imread(data_path + f"train/{data.id[i]}.tiff")

        if (img.shape[0] == 3):
            img = np.moveaxis(img, 0, 2)
        X_images.append(img)
        shape = (img.shape[1], img.shape[0])
        mask = enc2mask(data.loc[data.id == data.id[i], "encoding"].values[0], shape)
        Masks.append(mask)
        image_dims.append(img.shape)
        print(img.shape)

    del img, mask, shape
    gc.collect()
    positive_idxs, negative_idxs = get_indexes(config["val_index"], X_images, Masks, image_dims,
                                               config["size"], config["step_size"])
    kid_sampler = KidneySampler(positive_idxs, negative_idxs, config["not_empty_ratio"])
    train_dataset = KidneyLoader(X_images, Masks, image_dims, False, config["size"],
                                 step_size=config["step_size"], val_index=config["val_index"])
    val_dataset = KidneyLoader(X_images, Masks, image_dims, True, config["size"], val_index=config["val_index"])
    if not config["use_sampler"]:
        print("Use full dataset")
        trainloader = DataLoader(train_dataset, batch_size=config["bs"], shuffle=True, num_workers=16)
    else:
        print("Use sample dataset")
        trainloader = DataLoader(train_dataset, batch_size=config["bs"], shuffle=False, num_workers=16, sampler=kid_sampler)
    valloader = DataLoader(val_dataset, batch_size=config["bs"] * 2, shuffle=False, num_workers=16)
    x, y, key = train_dataset[10]
    print(x.shape, y.shape)
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1,
                     decoder_use_batchnorm=False).cuda()
    if config["loss_name"] == 'comboloss':
        print("Use combo loss")
        loss = ComboLoss(weights=config["weights"])
    else:
        print("Use BCE Loss")
        loss = BCEWithLogitsLoss()
    optim = AdamW(model.parameters(), lr=config["max_lr"])
    metric = smp.utils.losses.DiceLoss()
    SchedulerClass_cos = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_params_cos = dict(
        T_max=config["epochs"], eta_min=config["min_lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **scheduler_params_cos)
    # ---- Start train loop here ----- #
    print('Start train')
    min_val_loss = 100
    max_val_dice = -100
    for epoch in range(config["epochs"] + config["epochs_minlr"]):
        with open(f"../{model_name}/{model_name}.log", 'a+') as logger:
            logger.write(f"Epoch # {epoch}, lr = {optim.param_groups[0]['lr']}\n")
        empty_cache()
        dice = 0
        print(f"epoch number {epoch}, lr = {optim.param_groups[0]['lr']}")
        if (config["use_sampler"]&config["use_adaptive_sampler"]):
            if epoch < 10:
                kid_sampler = KidneySampler(positive_idxs, negative_idxs, 0.2)
                trainloader = DataLoader(train_dataset, batch_size=config["bs"],
                                         shuffle=False, num_workers=16, sampler=kid_sampler)
            else:

                kid_sampler = KidneySampler(positive_idxs, negative_idxs, 0.5)
                trainloader = DataLoader(train_dataset, batch_size=config["bs"],
                                         shuffle=False, num_workers=16, sampler=kid_sampler)

        model, optim, pred_keys, final_masks, \
        train_loss, train_dice = train_one_epoch(model, optim, trainloader, config["size"], loss)
        print(f"train loss = {train_loss}")

        model, optim, val_keys, val_masks, val_loss = val_one_epoch(model, optim, valloader, config["size"], loss)
        print(f"val loss = {val_loss}")

        if epoch < config["epochs"]:
            scheduler.step()
        val_mask11, val_dice = calculate_dice(Masks, val_keys, val_masks, config["val_index"], config["size"])
        if val_dice > max_val_dice:
            max_val_dice = val_dice
            save(model.state_dict(), f"../{model_name}/{model_name}_{epoch}.h5")
            save(model.state_dict(), f"../{model_name}/last_best_model.h5")

        #    dices = []
        #     for image_number in range(1, len(Masks)):
        #         mask11, dice = calculate_dice(pred_keys, final_masks, image_number, config["size"])
        #         dices.append(dice)
        # print("Dice on train macro ", dices, np.mean(dices))
        print("Dice on train micro ", train_dice)
        print(f"Dice on val (1 image) = {val_dice}")
        with open(f"../{model_name}/{model_name}.log", 'a+') as logger:
            logger.write(f'train loss = {train_loss}, train dice (micro) = {train_dice}\n')
            logger.write(f'validation loss = {val_loss}, val dice = {val_dice}\n')
            logger.write('\n')
        print("=====================")
    model.load_state_dict(load(f"{model_name}/last_best_model.h5"))
    val_keys, val_masks = predict_data(model, valloader, config["size"], True)
    val_mask11, val_dice = calculate_dice(val_keys, val_masks, config["val_index"], config["size"])
    print(f"Dice on val (1 image) with TTA = {val_dice}")

    sample_sub = pd.read_csv(data_path + 'sample_submission.csv')
    test_paths = sample_sub.id.values
    print('Start test')
    X_test_images = []
    for name in test_paths:
        img = tiff.imread(data_path + f"test/{name}.tiff")
        if (img.shape[0] == 3):
            img = np.moveaxis(img, 0, 2)
        X_test_images.append(img)
        print(img.shape)
    del img
    gc.collect()

    image_dims = [(31295, 40429), (14844, 31262), (38160, 42360), (26840, 49780), (36800, 43780)]
    test_dataset = ValLoader(X_test_images, config["size"])
    testloader = DataLoader(test_dataset, batch_size=config["bs"] * 2, shuffle=False, num_workers=16)
    model.load_state_dict(load(f'../{model_name}/last_best_model.h5'))
    test_masks, test_keys = predict_test(model, config["size"], testloader, True)
    all_enc = []
    for n in range(len(sample_sub)):
        img_n_keys = [(i, k) for i, k in enumerate(test_keys) if k[0] == n]
        mask = mask_from_keys_and_preds_test(img_n_keys, test_masks, n, config["size"])
        t = 0.5
        mask[mask < t] = 0
        mask[mask >= t] = 1
        enc = mask2enc(mask)
        all_enc.append(enc[0])
    sample_sub.predicted = all_enc
    sample_sub.to_csv(f'../{model_name}/{model_name}.csv', index=False)


config = {}
config['cross_val'] = False
config['use_adaptive_sampler'] = False
config['epochs_minlr'] = 0
config['val_index'] = 7
config['gpu_number'] = 3
config['loss_name'] = 'comboloss'
prefix = 'rnet34_gridsearch_'
for size_after_reshape in [320]:
    for use_sampler in [True]:
        for size in [320, 512, 1024, 1500]:
            for bs in [64]:
                for epochs in [40]:
                    for not_empty_ratio in [0.3, 0.5]:
                        for max_lr in [7e-4]:
                            for min_lr in [1e-6]:
                                for weights in [{"bce": 1, "dice": 0, "focal": 0},
                                                {"bce": 1, "dice": 1, "focal": 1},
                                                 {"bce": 2, "dice": 1, "focal": 2},
                                                {"bce": 1, "dice": 2, "focal": 1}]:
                                    for step_size in [320, 512, 1024]:
                                        config['size'] = size
                                        config['use_sampler'] = use_sampler
                                        config['bs'] = bs
                                        config['epochs'] = epochs
                                        config['not_empty_ratio'] = not_empty_ratio
                                        config['max_lr'] = max_lr
                                        config['min_lr'] = min_lr
                                        config['weights'] = weights
                                        s = 'w_'
                                        for weight in weights:
                                            s += str(weights[weight])
                                            s += '-'
                                        config['step_size'] = step_size
                                        model_name = f'{prefix}_{s}_{size}_{step_size}_{size_after_reshape}_{bs}_{epochs}_{config["val_index"]}'
                                        config['model_name'] = model_name
                                        train_all(config)