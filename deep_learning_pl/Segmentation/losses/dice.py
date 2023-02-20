from numpy import random
import torch
from torch import Tensor
import torch.nn.functional as F
# https://github.com/pytorch/pytorch/issues/1249

def dice_coeff_easy(pred, target, smooth=1e-6):
    num = pred.size(0)
    m1 = pred.view(num, -1)  #Flatten
    m2 = target.view(num, -1)  #Flatten
    intersection = (m1*m2).sum()

    return (2*intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def dice_coeff(input:Tensor, target:Tensor, reduce_batch_first: bool = False, smooth=1e-6):
    # average of dice
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor wihout batch dimension (shape{input.shape})')
    
    if input.dim() == 2 or reduce_batch_first:
        # flatten
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2*inter

        return (2*inter + smooth) / (sets_sum + smooth)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def multiclass_dice_coeff(input:Tensor, target: Tensor, reduce_batch_first:bool=True, smooth=1e-6):
    # average of dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, smooth)
    
    return dice / input.shape[1]

def dice_loss(input:Tensor, target: Tensor, multiclass: bool = True):
    # dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1-fn(input, target, reduce_batch_first=True)

def dice_c_plus(input:Tensor, target: Tensor, num_class, smooth=1e-6):
    target_onehot = torch.zeros_like(pred)   # N*C*H*W
    target_onehot.scatter_(1, target, 1)

    prediction_roi = pred.slice_scatter(1, 1, num_class, 1)
    target_roi = target_onehot.slice_scatter(1, 1, num_class, 1)
    intersection = (prediction_roi*target_roi).sum()
    union = prediction_roi.sum() + target_roi.sum() - intersection
    dice = (intersection + smooth) / (union + smooth)
    return dice

  
if __name__ == '__main__':
    pred = random.rand(3,3)
    truth = random.randint(0, 2, (3, 3)).astype('float64')
    pred =torch.from_numpy(pred)
    truth = torch.from_numpy(truth)

    loss = dice_coeff(pred, truth)
    print('torch loss is ', loss)
    closs = dice_c_plus(pred, truth, 2)
    print('cplus loss is ', closs)

