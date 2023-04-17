# A class for generating an image with multiple MNIST digits

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import random
class DatasetGenerator:

    def __init__(self):

        # Load MNIST dataset
        self.load_mnist()

    def load_mnist(self):
        # Load the MNIST dataset
        mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        self.mnist_data = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=True)
        self.images, self.targets = next(iter(self.mnist_data))


    def get_mnist_digit(self, digit):
        indices = (self.targets == digit).nonzero(as_tuple=False)
        return random.choice(self.images[indices]).squeeze().numpy()

    def create_combined_image(self, digits):
        digit_images = [Image.fromarray((self.get_mnist_digit(d) * 255).astype('uint8')) for d in digits]
        combined_image = Image.new('L', (28 * len(digits), 28))

        for i, img in enumerate(digit_images):
            combined_image.paste(img, (28 * i, 0))

        return combined_image

gen = DatasetGenerator()
digits_to_combine = [3, 5, 1, 7, 0, 0, 7]
combined_image = gen.create_combined_image(digits_to_combine)
combined_image.show()