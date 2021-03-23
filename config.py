gpu_number = 2
size_after_reshape = 320
cross_val = False
use_sampler = True
use_adaptive_sampler = False
size = 1024
bs = 64
epochs = 30
epochs_minlr = 0
not_empty_ratio = 0.5
val_index = [1,7,13]
loss_name = 'comboloss'
max_lr = 7e-4
min_lr = 3e-6
prefix = 'rnet34_combo'
weights = {"bce": 1, "dice": 0, "focal": 0}
s = 'w_'
for weight in weights:
    s += str(weights[weight])
    s += '-'
if cross_val:
    model_name = f'resize_cv_{s}_{size}_{bs}_{epochs}'
else:
    val_index_print = ''.join([str(x)+',' for x in val_index])
    model_name = f'{prefix}_{s}_{size}_{size_after_reshape}_{bs}_{epochs}_{val_index_print[:-1]}'
step_size = int(size*0.5)