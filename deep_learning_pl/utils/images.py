import warnings
from PIL import Image


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