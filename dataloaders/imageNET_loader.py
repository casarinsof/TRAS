from torch.utils.data.dataset import Dataset
import cv2
import lmdb
import pickle


class ImageNet(Dataset):
    def __init__(self, file_path, transform=None, *args, **kwargs):
        self.env = lmdb.open(file_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.stat()['entries'] - 1)
        self.transform = transform

    def _cv2_decode(self, data):
        return cv2.imdecode(data[0], cv2.IMREAD_COLOR), data[1]

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            pack = pickle.loads(txn.get(f'{index:0>8d}'.encode()))
        image, label = self._cv2_decode(pack)

        # image: uint8 0-255 image with VARIABLE width and heights
        # label: int 0-999 for image-1k annotations
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.length