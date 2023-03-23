import warnings
import torch
from PIL import Image
import matplotlib.pyplot as plt

__all__ = ["open_img"]


def open_img(path, mode=None):
    """
    Input: 
        path: img path
        mode: basic selection range is  [None, 'L', 'RGB']
    Output:
        img (np.ndarray)
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        img = Image.open(path)
        if mode and img.mode != mode:
            img = img.convert(mode)

    return img


def show_img_from_dataset(dataset, show_label=False):
    figure = plt.figure(figsize=(2, 2))
    cols, rows = 2, 2
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        res = dataset[sample_idx]
        img, label = res["image"], res['mask']
        figure.add_subplot(rows, cols, i)
        # plt.title(dataset[label])
        plt.axis("off")
        if show_label:
            plt.imshow(label.squeeze(), cmap="gray")
        else:
            plt.imshow(img.permute(1, 2, 0), cmap='gray')
    plt.show()
