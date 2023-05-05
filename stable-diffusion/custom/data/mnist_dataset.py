import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

# class MNISTBase(Dataset):
#     def __init__(self,
#                  indices,
#                  txt_file,
#                  data_root,
#                  size=None,
#                  ):
#         self.indices = indices
#         self.data_root = data_root
#         self._length = len(self.indices)
#         self.labels = {
#             "relative_file_path_": [l for l in self.image_paths],
#             "file_path_": [os.path.join(self.data_root, l)
#                            for l in self.image_paths],
#         }

#         self.size = size


#     def __len__(self):
#         return self._length

#     def __getitem__(self, i):
#         example = dict((k, self.labels[k][i]) for k in self.labels)
#         image = Image.open(example["file_path_"])
#         if not image.mode == "RGB":
#             image = image.convert("RGB")

#         # default to score-sde preprocessing
#         img = np.array(image).astype(np.uint8)
#         crop = min(img.shape[0], img.shape[1])
#         h, w, = img.shape[0], img.shape[1]
#         img = img[(h - crop) // 2:(h + crop) // 2,
#               (w - crop) // 2:(w + crop) // 2]

#         image = Image.fromarray(img)
#         if self.size is not None:
#             image = image.resize((self.size, self.size), resample=self.interpolation)

#         image = self.flip(image)
#         image = np.array(image).astype(np.uint8)
#         example["image"] = (image / 127.5 - 1.0).astype(np.float32)
#         return example


# class MNISTTrain(MNISTBase):
#     def __init__(self, **kwargs):
#         super().__init__(indicies=[i for i in range(80)], data_root="data/mnist", **kwargs)


# class MNISTValidation(MNISTBase):
#     def __init__(self, flip_p=0.0, **kwargs):
#         super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/mnist",
#                          flip_p=flip_p, **kwargs)

class MNISTDataset(Dataset):
    def __init__(self, root_dir, indices=None, transform=None, 
                 size=None,
                 interpolation="bicubic",
                 ):
        self.root_dir = root_dir
        self.all_image_files = os.listdir(root_dir)
        self.indices = indices if indices is not None else [i for i in range(len(self.all_image_files))]
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

        # Convert the image to RGB if it is grayscale
        if not image.mode == "RGB":
            image = image.convert("RGB")

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
        super().__init__('data/mnist_dataset/dataset', indices=[i for i in range(8000)], **kwargs)

class MNISTValidation(MNISTDataset):
    def __init__(self, **kwargs):
        super().__init__('data/mnist_dataset/dataset', indices=[i for i in range(8000, 10000)], **kwargs)

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