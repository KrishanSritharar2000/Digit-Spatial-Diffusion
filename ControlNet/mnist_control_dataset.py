import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pickle
import datetime
import matplotlib.pyplot as plt
import re

class MNISTControlDataset(Dataset):
    def __init__(self, transform=None, 
                 size=128,
                 interpolation="bicubic",
                 ):
        self.root_dir = '../stable-diffusion/data/mnist_dataset/dataset'
        self.all_image_files = os.listdir(self.root_dir)

        # if indicesFile is not None:
        #     with open(indicesFile, 'rb') as f:
        #         self.indices = pickle.load(f)
        # else:
        # self.indices = np.arange(len(self.all_image_files))
        with open('../stable-diffusion/data/train_indices.pkl', 'rb') as f:
            self.indices = pickle.load(f)

        self.transform = transform
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        
        self.relations = {
            "left of": 0,
            "right of": 1,
            "above": 2,
            "below": 3
        }

        self.digit_regex = r'\b\d\b'
        self.relationship_regex = r'\b(left of|right of|above|below)\b'


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
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

        #shift image values from [0,1] into [-1,1]
        image = ((image * 2.0) - 1.0).to(torch.float32)

        hint = self.convertLabelToHintTensor(label)

        return dict(jpg=image, txt=label, hint=hint)

    def convertLabelToHintTensor(self, label):
        #Make a tensor of size 10x10x4
        matrix = torch.zeros((10,10,4))
        # strip label of whistespaces
        label = label.strip()
        digits = re.findall(self.digit_regex, label)
        relationships = re.findall(self.relationship_regex, label)
        digits = list(map(int, digits))
        triplets = [(digits[i], relationships[i], digits[i + 1]) for i in range(len(relationships))]
        for triplet in triplets:
            matrix[triplet[0], triplet[2], self.relations[triplet[1]]] = 1
        matrix = matrix.reshape(1, 20, 20)
        return matrix




    
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
                    imgs.append(MNISTControlDataset.visualize(tensor[i]))
                return imgs
        return [MNISTControlDataset.visualize(tensor)]

    @staticmethod
    def denormaliseOneLayer(tensor):
        tensor = (tensor + 1.0) / 2.0  # scale to [0, 1] range
        tensor = tensor * 255
        tensor = tensor.clamp(0, 255)  # ensure values are within [0, 255]
        return tensor.to(torch.uint8)
    
    @staticmethod
    def visualiseAndSave(tensor, reconstructions, conds):
        input_tensor = tensor.cpu().numpy()  # If your tensor is on GPU, move it to CPU first and then convert to numpy array
        reconstructions_list = MNISTControlDataset.denormalise(reconstructions)
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))

                # Iterate through the images in the batch and display them in the subplots
        for i, ax in enumerate(axes.flat):
            if i % 2 == 0:
                image = input_tensor[i // 2, 0, :, :]
                ax.set_title(f'Input {i//2 + 1}: {conds[i//2]}')  # Set the title to include the label
            else:
                image = reconstructions_list[i // 2] #, 0, :, :]
                # ax.set_title(f'Reconstructed {i//2 + 1}: {batch["label"][i//2]}')  # Set the title to include the label
            ax.imshow(image, cmap='gray')
            ax.axis('off')
        # Display the grid of images
        plt.tight_layout()
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        plt.savefig(f"training_ldm_log/recon/{now}.png")
