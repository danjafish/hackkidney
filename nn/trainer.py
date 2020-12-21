from torch.nn.functional import interpolate
import torch
from torch import sigmoid, no_grad, save, load
from torch.cuda import empty_cache
from utils.support_func import dice_score, mask_tresholed
from dataset.dataloader import *


def train_one_epoch(model, optim, trainloader, size, loss):
    train_dice = 0
    final_masks = []
    pred_keys = []
    train_loss = 0
    model = model.train();
    for i, (x, y_true, key) in enumerate(trainloader):
        x = x.permute((0, 3, 1, 2)).cuda().float()
        y_true = y_true.cuda().float()
        y_pred = model(x)  # [:,0]
        pred_keys.extend((k1, k2, k3) for k1, k2, k3 in
                         zip(key[0].numpy(), key[1].numpy(), key[2].numpy()))
        big_masks = interpolate(y_pred, (size, size))
        big_ground_true = interpolate(y_true, (size, size))
        l = loss(y_pred, y_true)
        optim.zero_grad()
        l.backward()
        optim.step()

        for pred, true_mask in zip(torch.sigmoid(big_masks).detach().cpu().numpy(),
                                   big_ground_true.detach().cpu().numpy()):
            final_masks.append(pred)
            train_dice += dice_score(mask_tresholed(pred), true_mask) / x.shape[0]
        train_loss += l.item() / len(trainloader)
    #         if i%300 == 10:
    #             #print(i, train_loss)
    #             break
    return model, optim, pred_keys, final_masks, train_loss, train_dice / len(trainloader)


def val_one_epoch(model, optim, valloader, size, loss):
    val_masks = []
    val_keys = []
    val_loss = 0.0
    with torch.no_grad():
        model = model.eval();
        for i, (x, y_true, key) in enumerate(valloader):
            x = x.permute((0, 3, 1, 2))
            x = x.cuda().float()
            y_true = y_true.cuda().float()
            y_pred = model(x)  # [:,0]
            big_masks = interpolate(y_pred, (size, size))
            # big_ground_true = interpolate(y_true, (size, size))
            for pred in torch.sigmoid(big_masks).detach().cpu().numpy():
                val_masks.append(pred)

            val_keys.extend((k1, k2, k3) for k1, k2, k3 in
                            zip(key[0].numpy(), key[1].numpy(), key[2].numpy()))
            l = loss(y_pred, y_true)
            val_loss += l.item() / len(valloader)

    return model, optim, val_keys, val_masks, val_loss


def predict_data(model, loader, size, tta=False):
    val_masks = []
    val_keys = []
    with torch.no_grad():
        model = model.eval();
        for i, (x, y_true, key) in enumerate(loader):
            x = x.permute((0, 3, 1, 2))
            x = x.cuda().float()
            if not tta:
                y_pred = model(x)  # [:,0]
            else:
                s1 = model(x)
                s2 = model(torch.flip(x, [-1]))
                s3 = model(torch.flip(x, [-2]))

                y_pred = (torch.flip(s2, [-1]) + torch.flip(s3, [-2]) + s1) / 3
            big_masks = interpolate(y_pred, (size, size))
            for pred in torch.sigmoid(big_masks).detach().cpu().numpy():
                val_masks.append(pred)

            val_keys.extend((k1, k2, k3) for k1, k2, k3 in
                            zip(key[0].numpy(), key[1].numpy(), key[2].numpy()))
    return val_keys, val_masks


def predict_test(model, size, testloader, tta=False):
    val_masks = []
    val_keys = []
    val_loss = 0
    with no_grad():
        model = model.eval()
        for i, (x, key) in enumerate(testloader):
            x = x.permute((0, 3, 1, 2))
            x = x.cuda().float()
            if not tta:
                y_pred = model(x)  # [:,0]
            else:
                s1 = model(x)
                s2 = model(torch.flip(x, [-1]))
                s3 = model(torch.flip(x, [-2]))

                y_pred = (torch.flip(s2, [-1]) + torch.flip(s3, [-2]) + s1) / 3
            big_masks = interpolate(y_pred, (size, size))
            for pred in torch.sigmoid(big_masks).detach().cpu().numpy():
                val_masks.append(pred)
            val_keys.extend((k1, k2, k3) for k1, k2, k3 in
                            zip(key[0].numpy(), key[1].numpy(), key[2].numpy()))
    return val_masks, val_keys