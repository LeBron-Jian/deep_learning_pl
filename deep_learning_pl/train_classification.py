from data.split_train_val import train_val_test_split_func
from datasets.Caltech101_classify import Caltech101Dataset


# step1: load Dataset
root_path = r"D:\workdata\data\caltech-101"
dataset = Caltech101Dataset(root_path)
# step2: split train val
train_loader, val_loader, test_loader = train_val_test_split_func(dataset)
print(len(train_loader), len(val_loader), len(test_loader))

