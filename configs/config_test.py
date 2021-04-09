import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='efficientnet-b4')
    parser.add_argument('--bs', '-b', type=int, default=32,
                        help='Batch size per process (default: 32)')
    parser.add_argument('--seed', type=int, default=314159,
                        help='Random seed')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--gpu-number', type=str, default='4')
    parser.add_argument('--size-after-reshape', type=int, default=320)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--step-size-ratio', type=float, default=0.5)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--predict-by-epochs', type=str, default='all')
    parser.add_argument('--best-dice-epochs', nargs='+', required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--store-masks', dest='store_masks', action='store_true')
    parser.add_argument('--not-store-masks', dest='store_masks', action='store_false')
    parser.add_argument('--model-name', type=str, default='unet++')
    parser.add_argument('--cros-val', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    return parser.parse_args()


# size_after_reshape = 320
fp16 = False
new_augs = True
