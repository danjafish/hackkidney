import pandas as pd
from torch.cuda import empty_cache
import numpy as np
from torch.utils.data import DataLoader
from config import *
from utils.support_func import *
from nn.trainer import *
from nn.predicter import *
import segmentation_models_pytorch as smp
import gc
from dataset.dataloader import KidneySampler, KidneyLoader, ValLoader
import tifffile as tiff
from torch.optim import AdamW, Adam
from torch.nn import BCEWithLogitsLoss
import os
import apex


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    encoder = args.encoder
    bs = args.bs
    prefix = encoder
    max_lr = args.max_lr
    # fp16 = args.fp16
    min_lr = args.min_lr
    size = args.size
    size_after_reshape = args.size_after_reshape
    step_size_ratio = args.step_size_ratio
    step_size = int(size * step_size_ratio)
    gpu_number = args.gpu_number
    loss_weights = args.loss_weights
    store_masks = args.store_masks
    weights = {"bce": int(loss_weights[0]), "dice": int(loss_weights[1]), "focal": int(loss_weights[2])}

    for weight in weights:
        s += str(weights[weight])
        s += '-'
    if cross_val:
        model_name = f'resize_cv_{s}_{size}_{bs}_{epochs}'
    else:
        val_index_print = ''.join([str(x) + ',' for x in val_index])
        model_name = f'{prefix}_{new_augs}_{fp16}_{s}_{size}_{step_size}_{size_after_reshape}_{bs}_{epochs}_{val_index_print[:-1]}'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    print('Start data preparation')
    seed_everything(2021)
    os.system(f"mkdir ../{str(model_name)}")
    # data_path = '/home/data/Kidney/data/'
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
    train_dataset = KidneyLoader(X_images, Masks, image_dims, False, size, step_size=step_size,
                                 val_index=val_index, new_augs=new_augs, size_after_reshape=size_after_reshape)
    val_dataset = KidneyLoader(X_images, Masks, image_dims, True, size, val_index=val_index,
                               new_augs=new_augs, size_after_reshape=size_after_reshape)
    if not use_sampler:
        print("Use full dataset")
        trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=16)
    else:
        print("Use sample dataset")
        trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=16, sampler=kid_sampler)
    valloader = DataLoader(val_dataset, batch_size=bs * 2, shuffle=False, num_workers=16)
    x, y, key = train_dataset[10]
    print(len(trainloader), len(valloader))
    model = smp.Unet(encoder, encoder_weights="imagenet", in_channels=3, classes=1,
                            decoder_use_batchnorm=False).cuda()

    if loss_name == 'comboloss':
        print("Use combo loss")
        loss = ComboLoss(weights=weights)
    else:
        print("Use BCE Loss")
        loss = BCEWithLogitsLoss()
    optim = AdamW(model.parameters(), lr=max_lr)
    if fp16:
        model, optimizer = apex.amp.initialize(
            model,
            optim,
            opt_level='O1')
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
    best_dice_epochs = []
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
        del pred_keys, final_masks
        gc.collect()
        model, optim, val_keys, val_masks, val_loss = val_one_epoch(model, optim, valloader, size, loss)
        print(f"val loss = {val_loss}")

        if epoch < epochs:
            scheduler.step()
        m = 0
        for img_number in val_index:
            _, val_dice = calculate_dice(Masks, val_keys, val_masks, img_number, size, image_dims)
            with open(f"../{model_name}/{model_name}.log", 'a+') as logger:
                logger.write(f'dice on image {img_number} = {val_dice} ')
            print(f'dice on image {img_number} = {val_dice}')
            m += val_dice
        with open(f"../{model_name}/{model_name}.log", 'a+') as logger:
            logger.write(f'\n')
        val_dice = m / len(val_index)
        if predict_by_epochs != 'best':
            if len(best_dice_epochs) < predict_by_epochs:
                best_dice_epochs.append((epoch, val_dice))
                save(model.state_dict(), f"../{model_name}/{model_name}_{epoch}.h5")
            else:
                for ind, el in enumerate(best_dice_epochs):
                    if el[1] < val_dice:
                        best_dice_epochs[ind] = (epoch, val_dice)
                        save(model.state_dict(), f"../{model_name}/{model_name}_{epoch}.h5")
                        break
            print('Best epochs updated. ', best_dice_epochs)
        # val_dice = calc_average_dice(Masks, val_keys, val_masks, val_index, image_dims, size)
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
        del val_keys, val_masks
        gc.collect()
        print("=====================")
    with open(f"../{model_name}/{model_name}.log", 'a+') as logger:
        logger.write(f'Best epochs = {best_dice_epochs}\n')
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

    test_dataset = ValLoader(X_test_images, img_dims_test, size, new_augs=new_augs,
                             size_after_reshape=size_after_reshape)
    testloader = DataLoader(test_dataset, batch_size=bs * 2, shuffle=False, num_workers=16)
    if predict_by_epochs == 'best':
        model.load_state_dict(load(f'../{model_name}/last_best_model.h5'))
        test_masks, test_keys = predict_test(model, size, testloader, True)
        make_prediction(sample_sub, test_keys, test_masks, model_name, img_dims_test, size, t=0.4)
    else:
        bled_masks = [np.zeros(s[:2]) for s in img_dims_test]
        for epoch in best_dice_epochs:
            model.load_state_dict(load(f'../{model_name}/{model_name}_{epoch[0]}.h5'))
            test_masks, test_keys = predict_test(model, size, testloader, True)
            for n in range(len(sample_sub)):
                mask = make_masks(test_keys, test_masks, n, img_dims_test, size)
                bled_masks[n] += mask/len(best_dice_epochs)
        all_enc = []
        if store_masks:
            for j, mask in enumerate(bled_masks):
                np.savetxt(f'{model_name}/{model_name}_mask_{j}.txt', mask)
        for mask in bled_masks:
            t = 0.4
            mask[mask < t] = 0
            mask[mask >= t] = 1
            enc = mask2enc(mask)
            all_enc.append(enc[0])
        sample_sub.predicted = all_enc
        s = ''.join([str(e[0]) + '_' for e in best_dice_epochs])[:-1]
        sample_sub.to_csv(f'../{model_name}/mean_{model_name}_{s}.csv', index=False)

    # all_enc = []
    # for n in range(len(sample_sub)):
    #     img_n_keys = [(i, k) for i, k in enumerate(test_keys) if k[0] == n]
    #     mask = mask_from_keys_and_preds_test(img_n_keys, test_masks, n, img_dims_test, size)
    #     t = 0.4
    #     mask[mask < t] = 0
    #     mask[mask >= t] = 1
    #     enc = mask2enc(mask)
    #     all_enc.append(enc[0])
    # sample_sub.predicted = all_enc
    # sample_sub.to_csv(f'../{model_name}/best_{model_name}_{t}.csv', index=False)
    #
    # model.load_state_dict(load(f'../{model_name}/{model_name}_{epochs-1}.h5'))
    # test_masks, test_keys = predict_test(model, size, testloader, True)
    # all_enc = []
    # for n in range(len(sample_sub)):
    #     img_n_keys = [(i, k) for i, k in enumerate(test_keys) if k[0] == n]
    #     mask = mask_from_keys_and_preds_test(img_n_keys, test_masks, n, img_dims_test, size)
    #     t = 0.4
    #     mask[mask < t] = 0
    #     mask[mask >= t] = 1
    #     enc = mask2enc(mask)
    #     all_enc.append(enc[0])
    # sample_sub.predicted = all_enc
    # sample_sub.to_csv(f'../{model_name}/{model_name}_{epochs-1}_{t}.csv', index=False)
