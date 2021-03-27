import pandas as pd
from torch.utils.data import DataLoader
from configs.config_test import *
from dataset.dataloader import ValLoader
from utils.support_func import *
from nn.trainer import *
from nn.predicter import *
import segmentation_models_pytorch as smp
import gc
import tifffile as tiff
import os

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
    model_path = args.model_path
    overlap = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    print('Start data preparation')
    seed_everything(2021)
    data_path = '../data/'
    data = pd.read_csv(data_path + 'train.csv')
    model = smp.Unet(encoder, encoder_weights="imagenet", in_channels=3, classes=1,
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

    test_dataset = ValLoader(X_test_images, img_dims_test, size, new_augs=new_augs,
                             size_after_reshape=size_after_reshape, overlap=True, step_size=step_size)
    print(len(test_dataset))
    testloader = DataLoader(test_dataset, batch_size=bs * 2, shuffle=False, num_workers=16)
    if predict_by_epochs == 'best':
        model.load_state_dict(load(f'../{model_path}/last_best_model.h5'))
        test_masks, test_keys = predict_test(model, size, testloader, True)
        make_prediction(sample_sub, test_keys, test_masks,
                        model_path, img_dims_test, size, overlap, step_size, t=0.4)
    else:
        bled_masks = [np.zeros(s[:2]) for s in img_dims_test]
        for epoch in best_dice_epochs:
            model.load_state_dict(load(f'../{model_path}/{model_path}_{epoch}.h5'))
            test_masks, test_keys = predict_test(model, size, testloader, True)
            for n in range(len(sample_sub)):
                mask = make_masks(test_keys, test_masks, n, img_dims_test, size,
                                  overlap, step_size)
                bled_masks[n] += mask / len(best_dice_epochs)
        all_enc = []
        for mask in bled_masks:
            t = 0.4
            mask[mask < t] = 0
            mask[mask >= t] = 1
            enc = mask2enc(mask)
            all_enc.append(enc[0])
        sample_sub.predicted = all_enc
        s = ''.join([str(e[0]) + '_' for e in best_dice_epochs])[:-1]
        sample_sub.to_csv(f'../{model_path}/mean_{model_path}_{s}.csv', index=False)