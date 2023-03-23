import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets.caravan_dataset import BasicDataset, CarvanaDataset
from segmentation.losses.dice import dice_loss
from segmentation.utils.evaluate import evaluate
from segmentation.backbones.UNet_Series.Unet import UNet
# import segmentation_models_pytorch as smp


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.2,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create datasets loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = net(images)
                    if net.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        # true_masks = F.one_hot(true_masks.squeeze_(1), 2).permute(0, 3, 1, 2).float()
                        # print(masks_pred.shape, true_masks.shape)
                        # # torch.Size([4, 1, 320, 479]) torch.Size([4, 2, 320, 479])

                        loss += dice_loss(F.softmax(masks_pred, dim=1).float(),
                                          F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                          multiclass=True)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (1 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)
                        logging.info('Validation Dice score: {}'.format(val_score))

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


if __name__ == '__main__':
    load_path = None
    dir_img = Path(r'D:\Desktop\workdata\public\code_paper\Pytorch-UNet-master\data/imgs/')
    dir_mask = Path(r'D:\Desktop\workdata\public\code_paper\Pytorch-UNet-master\data/masks/')
    dir_checkpoint = Path('/mnt/shared/users/jianwang/code/Pytorch-UNet-master/datasets/checkpoints/')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # net = smp.UnetPlusPlus(
    #     encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
    #     in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    #     classes=2,  # model output channels (number of classes in your dataset)
    # )
    net = UNet(n_channels=3, n_classes=1)

    if load_path is not None:
        net.load_state_dict(torch.load(load_path, map_location=device))
        logging.info(f'Model loaded from {load_path}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=10,
                  batch_size=2,
                  learning_rate=1e-5,
                  device=device,
                  img_scale=0.25,
                  val_percent=0.5,
                  amp=False)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)



