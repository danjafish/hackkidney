import argparse

gpu_number = 4
size_after_reshape = 320
cross_val = False
use_sampler = True
use_adaptive_sampler = False
size = 1024
bs = 30
fp16 = False
epochs = 30
epochs_minlr = 0
not_empty_ratio = 0.5
val_index = [1, 7, 13]
loss_name = 'comboloss'
max_lr = 7e-4
min_lr = 3e-6
prefix = 'effb4'
predict_by_epochs = 'best'
weights_for_pred_epochs = 1 if predict_by_epochs == 'best' else [1]*len(predict_by_epochs)
new_augs = False
weights = {"bce": 1, "dice": 0, "focal": 0}
s = 'w_'
for weight in weights:
    s += str(weights[weight])
    s += '-'
if cross_val:
    model_name = f'resize_cv_{s}_{size}_{bs}_{epochs}'
else:
    val_index_print = ''.join([str(x)+',' for x in val_index])
    model_name = f'{prefix}_{new_augs}_{fp16}_{s}_{size}_{size_after_reshape}_{bs}_{epochs}_{val_index_print[:-1]}'
step_size = int(size*0.5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='efficientnet-b4')
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='Batch size per process (default: 32)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=7e-4,
                        metavar='LR',
                        help='Initial learning rate. Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256. A warmup schedule will also be applied over the first 5 epochs.')

    parser.add_argument('--seed', type=int, default=314159,
                        help='Random seed')

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--opt-level', type=str, default='O1')

    return parser.parse_args()
