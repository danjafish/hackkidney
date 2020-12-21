size_after_reshape = 256
cross_val = False
use_sampler = True
use_adaptive_sampler = False
size = 1024
bs = 64
epochs = 35
epochs_minlr = 0
not_empty_ratio = 0.5
val_index = 7
loss_name = 'comboloss'
max_lr = 1e-3
min_lr = 1e-6
prefix = 'test_git'
weights = {"bce": 1, "dice": 2, "focal": 1}
if cross_val:
    model_name = f'resize_cv_{loss_name}_{size}_{bs}_{epochs}'
else:
    model_name = f'{prefix}_{size}_{size_after_reshape}_{bs}_{epochs}_{val_index}'
step_size = int(size*0.75)

