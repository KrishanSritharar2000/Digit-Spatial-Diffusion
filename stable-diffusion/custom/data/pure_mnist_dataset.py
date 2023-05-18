import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

#Was going to be used to test normal mnist digits when the autoencder wasnt working
class PureMNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None, size=None):
        self.root = root
        self.train = train
        self.transform = transform

        # Load MNIST dataset
        self.data = datasets.MNIST(root=self.root, train=self.train, download=True)

    def __len__(self):
        # Return the total number of data samples
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()

        # Retrieve the image and label at the given index
        image, label = self.data[index]

        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # Apply transformations to the image (if any)
        if self.transform is not None:
            image = self.transform(image)

        output = {}
        output["image"] = image
        output["caption"] = label
        return output
    

class PureMNISTTrain(PureMNISTDataset):
    def __init__(self, **kwargs):
        super().__init__('data/', **kwargs)

class PureMNISTValidation(PureMNISTDataset):
    def __init__(self, **kwargs):
        super().__init__('data/', **kwargs)

class PureMNISTTest(PureMNISTDataset):
    def __init__(self, **kwargs):
        super().__init__('data/mnist_dataset/test_dataset', **kwargs)