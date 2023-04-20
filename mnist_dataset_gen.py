# A class for generating an image with multiple MNIST digits

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import random
class DatasetGenerator:

    def __init__(self, rows=3, cols=3):
        assert rows == cols, 'Only square images are supported'
        self.rows = rows
        self.cols = cols

        # self.spatial_relationships = {
        #     'left': ['to the left of', 'left of', 'on the left of'],
        #     'right': ['to the right of', 'right of', 'on the right of'],
        #     'above': ['above', 'on top of', 'on the top of', 'over'],
        #     'below': ['below', 'under', 'underneath', 'beneath'],
        # }

        self.spatial_relationships = {
            'left': ['left of'],
            'right': ['right of'],
            'above': ['above'],
            'below': ['below'],
        }

        # Load MNIST dataset
        # self.load_mnist()

    def load_mnist(self):
        # Load the MNIST dataset
        mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        self.mnist_data = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=True)
        self.images, self.targets = next(iter(self.mnist_data))


    def get_mnist_digit(self, digit):
        indices = (self.targets == digit).nonzero(as_tuple=False)
        return random.choice(self.images[indices]).squeeze().numpy()

    def create_combined_image(self, digits):
        assert len(digits) <= self.rows * self.cols, 'Number of digits be less than the number of cells in the image'

        chosen_digits = [self.get_mnist_digit(d) for d in digits]
        digit_images = [Image.fromarray((d * 255).astype('uint8')) for d in chosen_digits]
        combined_image = Image.new('L', (28 * self.cols, 28 * self.rows))

        for i, img in enumerate(digit_images):
            row = i // self.rows
            col = i % self.cols
            combined_image.paste(img, (28 * col, 28 * row))

        return combined_image
    

    def calculate_displacement(self, relationships):
        displacement = [0, 0]
        for relationship in relationships:
            if relationship == 'left':
                displacement[0] -= 1
            elif relationship == 'right':
                displacement[0] += 1
            elif relationship == 'above':
                displacement[1] -= 1
            elif relationship == 'below':
                displacement[1] += 1
        return displacement

    def calculate_start_position(self, displacement):
        start_row = (self.rows - 1) // 2 + displacement[1]
        start_col = (self.cols - 1) // 2 + displacement[0]
        return start_row, start_col

    def create_prompt_image(self, prompt):
        prompt_str, digits, relationships = prompt
        displacement = self.calculate_displacement(relationships)
        start_row, start_col = self.calculate_start_position(displacement)
        
        prompt_image = Image.new('L', (28 * self.cols, 28 * self.rows))

        # put the digits in the right postions on the grid accordings to the relationships
        for i, digit in enumerate(digits):
            row = i // self.rows
            col = i % self.cols
            digit_image = Image.fromarray((self.get_mnist_digit(digit) * 255).astype('uint8'))
            prompt_image.paste(digit_image, (28 * col, 28 * row))

        

    def create_a_prompt(self, num_digits):
        digits = []
        relationships = []
        prompt = ""
        for i in range(num_digits - 1):
            # select a random digit from 1 to 10
            digit = random.randint(1, 10)
            digits.append(digit)
            
            # select a random spatial relationship
            spatial_relationship = random.choice(list(self.spatial_relationships.keys()))
            relationships.append(spatial_relationship)
            spatial_relationship = self.spatial_relationships[spatial_relationship]

            prompt += f"{digit} {random.choice(spatial_relationship)} "            
        digit = random.randint(1, 10)
        prompt += f"{digit}"

        return prompt, digits, relationships

gen = DatasetGenerator()
digits_to_combine = [3, 5, 1, 7, 0, 0, 7]
prompt = gen.create_a_prompt(3)
gen.create_prompt_image(prompt)
print(prompt)
# combined_image = gen.create_combined_image(digits_to_combine)
# combined_image.show()