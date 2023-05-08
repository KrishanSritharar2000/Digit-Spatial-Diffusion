import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pickle
class MNISTDataset(Dataset):
    def __init__(self, root_dir, indicesFile=None, transform=None, 
                 size=None,
                 interpolation="bicubic",
                 ):
        self.root_dir = root_dir
        self.all_image_files = os.listdir(root_dir)

        if indicesFile is not None:
            with open(indicesFile, 'rb') as f:
                self.indices = pickle.load(f)

        self.transform = transform
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        output = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.all_image_files[self.indices[idx]])
        image = Image.open(img_path)

        # # Convert the image to RGB if it is grayscale
        # if not image.mode == "RGB":
        #     image = image.convert("RGB")

        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)


        # Extract the label from the filename, assuming the format "{label}_*.jpg" or "{label}_*.png"
        label = self.all_image_files[self.indices[idx]].split("_")[1]
        
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        if self.transform:
            image = self.transform(image)

        normalized_image = (image / 127.5 - 1.0).to(torch.float32)

        output["image"] = normalized_image
        output["label"] = label
        return output

class MNISTTrain(MNISTDataset):
    def __init__(self, **kwargs):
        super().__init__('data/mnist_dataset/dataset', indicesFile="data/train_indices.pkl", **kwargs)

class MNISTValidation(MNISTDataset):
    def __init__(self, **kwargs):
        super().__init__('data/mnist_dataset/dataset', indicesFile="data/test_indices.pkl", **kwargs)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    dataset = MNISTTrain(transform=transform)
    print(len(dataset))
    print(dataset[0]["image"].shape)
    print(dataset[0]["label"])
    plt.imshow(dataset[0]["image"].squeeze().numpy())
    plt.show()