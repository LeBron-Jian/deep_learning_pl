from data.split_train_val import train_val_test_split_func
from datasets.Caltech101_classify import Caltech101Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import optim
from PIL import Image
from tqdm import tqdm
import torch
import timm
import logging
from pathlib import Path
import torch.nn as nn
import matplotlib.pyplot as plt

# step1: load Dataset
root_path = r"D:\workdata\data\caltech-101"
dataset = Caltech101Dataset(root_path)

# step2: split train val
# set transforms
transforms_train = transforms.Compose([
    transforms.Resize((230, 230)),
    transforms.RandomRotation(15, ),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])
transforms_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])
train_dataset, val_dataset, test_dataset = train_val_test_split_func(dataset,
                                                                     train_transforms=transforms_train,
                                                                     val_transforms=transforms_valid,
                                                                     test_transforms=transforms_valid)

# step3  create dataloader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# stpe4 load model
net = timm.create_model('vgg16', pretrained=True)

# step5. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
learning_rate = 1e-5
amp = False  # Mixed Precision
optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
criterion = nn.CrossEntropyLoss()
global_step = 0

# step6. Begin training
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_checkpoint = True
dir_checkpoint = Path('./checkpoints/')
for epoch in range(epochs):
    epoch_loss = 0
    with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for i, batch in enumerate(train_loader, 0):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            # images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # true_masks = true_masks.to(device=device, dtype=torch.long)

            # assert images.shape[1] == net.n_channels, \
            #     f'Network has been defined with {net.n_channels} input channels, ' \
            #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
            #     'the images are loaded correctly.'

            optimizer.zero_grad(set_to_none=True)
            net = net.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.update(inputs.shape[0])  # 更新进度条
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

    # Evaluation round
    # division_step = (len(train_dataset) // (1 * batch_size))
    # if division_step > 0:
    #     if global_step % division_step == 0:
    #         val_score = evaluate(net, val_loader, device)
    #         scheduler.step(val_score)
    #         logging.info('Validation Dice score: {}'.format(val_score))

    if save_checkpoint:
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
        logging.info(f'Checkpoint {epoch + 1} saved!')
