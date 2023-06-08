import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.class_voc_dataset import PascalVOCDataset
from torchvision.transforms import transforms


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


Voc_data_path = r"D:\workdata\data\VOC\VOC2012"
train_data = PascalVOCDataset(Voc_data_path, transforms=data_transforms)
val_data = PascalVOCDataset(Voc_data_path, transforms=data_transforms)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0)


# defination model
class VOCClassifier(nn.Module):
    def __init__(self, num_classes=21):
        super(VOCClassifier, self).__init__()
        self.backbone = timm.create_model('vgg16', pretrained=True)
        self.backbone.head.fc = nn.Linear(self.backbone.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VOCClassifier().to(device)


# criterion = nn.CrossEntropyLoss()
# criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

print('Training complete.')

# TODO   注意多分类，比如一张图有多个标签，如何做分类。！！
# 一张图片只有一个标签！！！ 又怎么做
# 对于标签进行one-hot



