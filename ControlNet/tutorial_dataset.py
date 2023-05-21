import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']


        source = cv2.imread('./training/fill50k/' + source_filename, 0)
        target = cv2.imread('./training/fill50k/' + target_filename, 0)

        # Do not forget that OpenCV read images in BGR order.
        #Read the images in grayscale
        # source = cv2.cvtColor(source, cv2.IMREAD_GRAYSCALE)
        # target = cv2.cvtColor(target, cv2.IMREAD_GRAYSCALE)
        # source = cv2.imread(source, 0)
        # target = cv2.imread(target, 0)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

