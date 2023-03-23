import os
import shutil
from tqdm import tqdm
from urllib.request import urlretrieve


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b*bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

# download dataset
def download(root):
    # load images
    filepath = os.path.join(root, "images.tar.gz")
    download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
    )

    extract_archive(filepath)

    # load annotations
    filepath = os.path.join(root, "annotations.tar.gz")
    download_url(
        url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
        filepath=filepath,
    )
    extract_archive(filepath)


if __name__ == "__main__":
    root = r'D:\Desktop\workdata\data'
    download(root)