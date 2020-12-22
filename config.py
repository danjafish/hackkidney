size_after_reshape = 256
cross_val = False
use_sampler = True
use_adaptive_sampler = False
size = 1024
bs = 64
epochs = 40
epochs_minlr = 0
not_empty_ratio = 0.5
val_index = 7
loss_name = 'bce'
max_lr = 1e-3
min_lr = 1e-6
prefix = 'param_test_bce'
weights = {"bce": 1, "dice": 0, "focal": 0}
s = 'w_'
for weight in weights:
    s+=str(weights[weight])
    s+='-'
if cross_val:
    model_name = f'resize_cv_{s}_{size}_{bs}_{epochs}'
else:
    model_name = f'{prefix}_{s}_{size}_{size_after_reshape}_{bs}_{epochs}_{val_index}'
step_size = int(size*0.5)

