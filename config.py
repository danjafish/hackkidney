import argparse


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


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
    parser.add_argument('--loss-weights', nargs='+', required=False)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--store-masks', dest='store_masks', action='store_true')
    feature_parser.add_argument('--not-store-masks', dest='store_masks', action='store_false')
    parser.add_argument('--cutmix', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.set_defaults(store_masks=False)
    parser.set_defaults(loss_weights=[1, 3, 1])
    return parser.parse_args()


# size_after_reshape = 320
use_sampler = True
use_adaptive_sampler = False
# size = 1024
# bs = 32
fp16 = False
# epochs = 30
#store_masks = False
epochs_minlr = 0
not_empty_ratio = 0.5
val_index = [1, 7, 13]
loss_name = 'comboloss'
#max_lr = 7e-4
#min_lr = 3e-6
predict_by_epochs = 4
# weights_for_pred_epochs = 1 if predict_by_epochs == 'best' else [1]*len(predict_by_epochs)
new_augs = True
#weights = {"bce": 1, "dice": 3, "focal": 1}
s = 'w_'
# step_size = int(size*0.5)
