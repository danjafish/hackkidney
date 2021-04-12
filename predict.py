import pandas as pd
from torch.utils.data import DataLoader
from configs.config_test import *
from dataset.dataloader import ValLoader
from torch.nn import DataParallel
from utils.support_func import *
from nn.trainer import *
from nn.predicter import *
import segmentation_models_pytorch as smp
import gc
import tifffile as tiff
import os
import h5py

if __name__ == '__main__':
    args = parse_args()
    encoder = args.encoder
    bs = args.bs
    prefix = encoder
    thr = args.thr
    size = args.size
    predict_by_epochs = args.predict_by_epochs
    best_dice_epochs = [int(x) for x in args.best_dice_epochs]
    fold = args.fold
    size_after_reshape = args.size_after_reshape
    step_size_ratio = args.step_size_ratio
    step_size = int(size * step_size_ratio)
    gpu_number = args.gpu_number
    model_path = args.model_path
    store_masks = args.store_masks
    cros_val = args.cros_val
    model_name = args.model_name
    parallel = args.parallel
    overlap = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    print('Start data preparation')
    seed_everything(2021)
    data_path = '../data/'
    data = pd.read_csv(data_path + 'train.csv')
    if model_name == 'unet':
        model = smp.Unet(encoder, encoder_weights="imagenet", in_channels=3, classes=1,
                         decoder_use_batchnorm=False).cuda()
    elif model_name == 'unet++':
        model = smp.UnetPlusPlus(encoder, encoder_weights="imagenet", in_channels=3, classes=1,
                                 decoder_use_batchnorm=False).cuda()
    else:
        print('Model name is incorrect. Set to unet++')
        model = smp.UnetPlusPlus(encoder, encoder_weights="imagenet", in_channels=3, classes=1,
                                 decoder_use_batchnorm=False).cuda()
    if parallel:
        model = DataParallel(model).cuda()

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

    test_dataset = ValLoader(X_test_images, img_dims_test, size, new_augs=new_augs,
                             size_after_reshape=size_after_reshape, overlap=True, step_size=step_size)
    print(len(test_dataset))
    testloader = DataLoader(test_dataset, batch_size=bs * 2, shuffle=False, num_workers=16)
    if predict_by_epochs == 'best':
        model.load_state_dict(load(f'../{model_path}/last_best_model.h5'))
        test_masks, test_keys = predict_test(model, size, testloader, True)
        make_prediction(sample_sub, test_keys, test_masks,
                        model_path, img_dims_test, size, overlap,
                        step_size, t=thr, store_masks=store_masks)
    else:
        bled_masks = [np.zeros(s[:2]) for s in img_dims_test]
        for epoch in best_dice_epochs:
            print(f'Predict by epoch {epoch}')
            if not cros_val:
                model.load_state_dict(load(f'../{model_path}/{model_path}_{epoch}.h5'))
            else:
                model.load_state_dict(load(f'../{model_path}/{model_path}_{epoch}_{fold}.h5'))
            test_masks, test_keys = predict_test(model, size, testloader, True)
            print(f'Start make masks for epoch {epoch}')
            for n in range(len(sample_sub)):
                mask = make_masks(test_keys, test_masks, n, img_dims_test, size,
                                  overlap, step_size)
                bled_masks[n] += np.round(mask, 4) / len(best_dice_epochs)
        all_enc = []
        del X_test_images
        gc.collect()
        print('Save masks')
        if store_masks:
            for j, mask in enumerate(bled_masks):
                with h5py.File(f'../{model_path}/{model_path}_mask_{j}.txt', "w") as f:
                    dset = f.create_dataset("mask", data=mask, dtype='f')

        print('Start making sub')
        for mask in bled_masks:
            t = thr
            mask[mask < t] = 0
            mask[mask >= t] = 1
            enc = mask2enc(mask)
            all_enc.append(enc[0])
        sample_sub.predicted = all_enc
        s = ''.join([str(e) + '_' for e in best_dice_epochs])[:-1]
        if not cros_val:
            sample_sub.to_csv(f'../{model_path}/mean_{model_path}_{s}_overlap_{overlap}.csv', index=False)
        else:
            sample_sub.to_csv(f'../{model_path}/mean_{model_path}_fold_{fold}_{s}_overlap_{overlap}.csv', index=False)