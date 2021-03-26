from utils.support_func import mask_from_keys_and_preds_test, mask2enc
from __main__ import size


def make_prediction(sample_sub, test_keys, test_masks, model_name, img_dims_test, t=0.4):
    all_enc = []
    for n in range(len(sample_sub)):
        img_n_keys = [(i, k) for i, k in enumerate(test_keys) if k[0] == n]
        mask = mask_from_keys_and_preds_test(img_n_keys, test_masks, n, img_dims_test, size)
        mask[mask < t] = 0
        mask[mask >= t] = 1
        enc = mask2enc(mask)
        all_enc.append(enc[0])
    sample_sub.predicted = all_enc
    sample_sub.to_csv(f'../{model_name}/best_{model_name}_{t}.csv', index=False)


def make_masks(test_keys, test_masks, n, img_dims_test):
    img_n_keys = [(i, k) for i, k in enumerate(test_keys) if k[0] == n]
    mask = mask_from_keys_and_preds_test(img_n_keys, test_masks, n, img_dims_test, size)
    return mask
