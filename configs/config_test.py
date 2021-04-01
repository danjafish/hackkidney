import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='efficientnet-b4')
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--bs', '-b', type=int, default=32,
                        help='Batch size per process (default: 32)')
    parser.add_argument('--max-lr', type=float, default=7e-4,
                        metavar='LR',
                        help='Initial learning rate.')

    parser.add_argument('--min-lr', type=float, default=3e-6,
                        metavar='LR',
                        help='Min learning rate.')
    parser.add_argument('--seed', type=int, default=314159,
                        help='Random seed')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--gpu-number', type=str, default='4')
    parser.add_argument('--size-after-reshape', type=int, default=320)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--step-size-ratio', type=float, default=0.5)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--predict-by-epochs', type=str, default='all')
    parser.add_argument('--best-dice-epochs', nargs='+', required=True)
    parser.add_argument('--store-masks', dest='store_masks', action='store_true')
    parser.add_argument('--not-store-masks', dest='store_masks', action='store_false')
    return parser.parse_args()


# size_after_reshape = 320
cross_val = False
use_sampler = True
use_adaptive_sampler = False
# size = 1024
# bs = 32
fp16 = False
# epochs = 30
epochs_minlr = 0
not_empty_ratio = 0.5
val_index = [1, 7, 13]
loss_name = 'comboloss'
max_lr = 7e-4
min_lr = 3e-6
# predict_by_epochs = 4
# weights_for_pred_epochs = 1 if predict_by_epochs == 'best' else [1]*len(predict_by_epochs)
new_augs = True
weights = {"bce": 1, "dice": 0, "focal": 0}
s = 'w_'
# best_dice_epochs = [0,1,2,3]
# step_size = int(size*0.5)
