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
        else:
            self.indices = np.arange(len(self.all_image_files))

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
        #Remove .png extension
        label = label.split(".")[0]
        
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        # to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        if self.transform:
            image = self.transform(image)

#float 32?
        #shift image values from [0,1] into [-1,1]
        image = ((image * 2.0) - 1.0).to(torch.float32)

        output["image"] = image
        output["caption"] = label
        return output
    
    @staticmethod
    def visualize(tensor):
        #reverse this (image * 2.0) - 1.0
        tensor = (tensor + 1.0) / 2.0  # scale to [0, 1] range
        denormalized = tensor * 255
        denormalized = denormalized.clamp(0, 255)  # ensure values are within [0, 255]
        # Convert back to PIL Image
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(tensor)
        return pil_img
    
    @staticmethod
    def denormalise(tensor):
        if (len(tensor.shape) == 4):
            if (tensor.shape[0] != 1):
                imgs = []
                for i in range(tensor.shape[0]):
                    imgs.append(MNISTDataset.visualize(tensor[i]))
                return imgs
        return [MNISTDataset.visualize(tensor)]

    @staticmethod
    def denormaliseOneLayer(tensor):
        tensor = (tensor + 1.0) / 2.0  # scale to [0, 1] range
        tensor = tensor * 255
        tensor = tensor.clamp(0, 255)  # ensure values are within [0, 255]
        return tensor.to(torch.uint8)



class MNISTTrain(MNISTDataset):
    def __init__(self, **kwargs):
        super().__init__('data/mnist_dataset/dataset', indicesFile="data/train_indices.pkl", **kwargs)

class MNISTValidation(MNISTDataset):
    def __init__(self, **kwargs):
        super().__init__('data/mnist_dataset/dataset', indicesFile="data/test_indices.pkl", **kwargs)

class MNISTTest(MNISTDataset):
    def __init__(self, **kwargs):
        super().__init__('data/mnist_dataset/test_dataset', **kwargs)


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