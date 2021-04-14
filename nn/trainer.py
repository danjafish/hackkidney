from torch.nn.functional import interpolate
import torch
from torch import sigmoid, no_grad, save, load
from utils.support_func import dice_score, mask_tresholed
from config import fp16
import apex
from torch.nn import BCEWithLogitsLoss
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


def train_one_epoch(model, optim, trainloader, size, loss, store_train_masks=True):
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
        if fp16:
            with apex.amp.scale_loss(l, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            l.backward()
        optim.step()

        for pred, true_mask in zip(torch.sigmoid(big_masks).detach().cpu().numpy(),
                                   big_ground_true.detach().cpu().numpy()):
            if store_train_masks:
                final_masks.append(pred)
            train_dice += dice_score(mask_tresholed(pred), true_mask) / x.shape[0]
        train_loss += l.item() / len(trainloader)
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


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False, freeze_encoder=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        for layer in self.encoder.parameters():
            layer.requires_grad = not freeze_encoder

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out