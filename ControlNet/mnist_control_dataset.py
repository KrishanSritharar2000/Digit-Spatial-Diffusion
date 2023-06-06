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
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
import warnings


class MNISTControlDataset(Dataset):
    def __init__(self, root_dir, indicesFile=None, transform=None, 
                 size=128,
                 interpolation="bicubic",
                 ):
        self.root_dir = root_dir
        self.all_image_files = os.listdir(self.root_dir)

        if indicesFile is not None:
            with open(indicesFile, 'rb') as f:
                self.indices = pickle.load(f)
        else:
            self.indices = np.arange(len(self.all_image_files))
        # if indicesFile is not None:
        #     with open(indicesFile, 'rb') as f:
        #         self.indices = pickle.load(f)
        # else:
        # self.indices = np.arange(len(self.all_image_files))
        # with open('../stable-diffusion/data/train_indices.pkl', 'rb') as f:
        #     self.indices = pickle.load(f)

        self.transform = transform
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.prompts = []



    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.all_image_files[self.indices[idx]])
        image = Image.open(img_path)

        # Extract the label from the filename, assuming the format "{label}_*.jpg" or "{label}_*.png"
        filename = self.all_image_files[self.indices[idx]].split("_")
        # self.prompts.append(self.all_image_files[self.indices[idx]].split(".")[0])
        label = filename[1]
        grid = filename[2]
        #Remove .png extension
        grid = grid.split(".")[0]
        grid = list(grid)
        
        # # Convert the image to RGB if it is grayscale
        # if not image.mode == "RGB":
        #     image = image.convert("RGB")
        hint = self.create_control_image(grid)

        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            hint = hint.resize((self.size, self.size), resample=self.interpolation)



        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        # to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        hint = to_tensor(hint)

        if self.transform:
            image = self.transform(image)

        #shift image values from [0,1] into [-1,1]
        image = ((image * 2.0) - 1.0).to(torch.float32)
        hint = ((hint * 2.0) - 1.0).to(torch.float32)
        # hint = self.convertLabelToHintTensor(label)



        return dict(jpg=image, txt=label, hint=hint)

    @staticmethod
    def convertLabelToHintTensor(label):
        relations = {
            "left of": 0,
            "right of": 1,
            "above": 2,
            "below": 3
        }

        digit_regex = r'\b\d\b'
        relationship_regex = r'\b(left of|right of|above|below)\b'


        #Make a tensor of size 10x10x4
        matrix = torch.zeros((10,10,4))
        # strip label of whistespaces
        label = label.strip()
        digits = re.findall(digit_regex, label)
        relationships = re.findall(relationship_regex, label)
        digits = list(map(int, digits))
        triplets = [(digits[i], relationships[i], digits[i + 1]) for i in range(len(relationships))]
        for triplet in triplets:
            matrix[triplet[0], triplet[2], relations[triplet[1]]] = 1
        matrix = rearrange(matrix, "h w c -> c h w")
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

    @staticmethod
    def create_control_image(grid, tensor=False, normalise=False, resize=False, size=128):
        warnings.filterwarnings("ignore")

        # Define the size of the grid and individual cell size
        grid_size = 3
        cell_size = 28

        # Create a blank image for the grid
        image_width = grid_size * cell_size
        image_height = grid_size * cell_size
        image = Image.new("L", (image_width, image_height), 0)
        draw = ImageDraw.Draw(image)

        # Define the font and font size
        font_size = 20
        font = ImageFont.truetype("arial.ttf", font_size)

        # Draw the digits on the grid
        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j
                digit = str(grid[index])
                if digit != "-":
                    digit = '.'
                    digit_width, digit_height = draw.textsize(digit, font=font)
                    x = j * cell_size + (cell_size - digit_width) // 2
                    y = i * cell_size + (cell_size - digit_height) // 2
                    draw.text((x, y), digit, fill=255, font=font)

        # Save the image
        # image.save("grid_image.png")
        if (resize):
            image = transforms.Resize((size, size))(image)
        if (tensor):
            image = transforms.ToTensor()(image)
        if (normalise):
            image = ((image * 2.0) - 1.0).to(torch.float32)
        return image
    
    def savePrompts(self, path):
        with open(path, 'w') as f:
            for item in self.prompts:
                f.write("%s\n" % item)

class MNISTControlTrain(MNISTControlDataset):
    def __init__(self, **kwargs):
        # super().__init__('../stable-diffusion/data/mnist_dataset/dataset', indicesFile="../stable-diffusion/data/train_indices.pkl", **kwargs)
        super().__init__('../data/new_w_grid_pos/train_dataset', indicesFile="../stable-diffusion/data/train_indices.pkl", **kwargs)

class MNISTControlValidation(MNISTControlDataset):
    def __init__(self, **kwargs):
        # super().__init__('../stable-diffusion/data/mnist_dataset/dataset', indicesFile="../stable-diffusion/data/test_indices.pkl", **kwargs)
        super().__init__('../data/new_w_grid_pos/train_dataset', indicesFile="../stable-diffusion/data/test_indices.pkl", **kwargs)


class MNISTControlTest(MNISTControlDataset):
    def __init__(self, **kwargs):
        # super().__init__('../stable-diffusion/data/mnist_dataset/test_dataset', **kwargs)
        super().__init__('../data/new_w_grid_pos/test_dataset', **kwargs)


if __name__ == "__main__":
    m = MNISTControlTest()
    for i in range(len(m)):
        m[i]
    m.savePrompts("control_test_prompts.txt")
    m.create_control_image([1,2,3,4,5,6,7,8,9])