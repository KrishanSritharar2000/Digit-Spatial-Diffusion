# Class to take in 84x84 image representing a 3x3 grid of MNIST digits and finds 
# all the spatial relationships between them. 
# The relationships include "left of", "right of", "above", "below",
import PIL
from PIL import Image
import numpy as np
from mnist_classifier import MNISTClassifier, CustomTensorDataset
import re

class DigitRelationshipFinder:
    def __init__(self, classifier):
        self.classifier = classifier
        self.digit_regex = r'\b\d\b'
        self.relationship_regex = r'\b(left of|right of|above|below)\b'

    def find_relationships(self, image):
        # Convert PIL Image to numpy array
        image = np.array(image)

        # Split the image into 9 digits
        all_digits = [Image.fromarray(image[28*(i//3):28*((i//3)+1), 28*(i%3):28*((i%3)+1)]) for i in range(9)]

        # Identify each digit using the classifier
        identified_digits = [self.classifier.classify_digit(digit) for digit in all_digits]

        relationships = []
        digits = list(filter(lambda x: x != 10, identified_digits))
        # Find the relationships
        for i, digit in enumerate(identified_digits):
            if digit == 10:
                continue
            # Compute the position in the 3x3 grid
            row_i, col_i = divmod(i, 3)

            # Now compare this digit to all other digits
            for j, other_digit in enumerate(identified_digits):
                if other_digit == 10 or i == j:
                    continue
                # Compute the position in the 3x3 grid
                row_j, col_j = divmod(j, 3)

                # Determine the relationship
                if col_j > col_i:
                    relationships.append((digit, "left of", other_digit))
                if col_j < col_i:
                    relationships.append((digit, "right of", other_digit))
                if row_j > row_i:
                    relationships.append((digit, "above", other_digit))
                if row_j < row_i:
                    relationships.append((digit, "below", other_digit))

        return set(relationships), digits
    
    def compute_accuracy(self, prompt, image):
        relationships, digits = self.find_relationships(image)
        prompt = prompt.strip()
        p_digits = re.findall(self.digit_regex, prompt)
        p_relationships = re.findall(self.relationship_regex, prompt)
        p_relationships_and_digits = [(p_digits[i], p_relationships[i], p_digits[i + 1]) for i in range(len(p_relationships))]
        accuracy = 0
        if len(digits) != len(p_digits):
            return 0
        for relationship in p_relationships_and_digits:
            if relationship in relationships:
                accuracy += 1


if __name__ == "__main__":
    # Load the classifier

    classifier = MNISTClassifier()
    classifier.load()
    
    # Load the image
    image = Image.open("samples/00000.png")
    size =  84 # 28*3
    image = image.resize((84, 84), resample=PIL.Image.BICUBIC)

    finder = DigitRelationshipFinder(classifier)
    relationships = finder.find_relationships(image)
    



    # # Print the relationships
    # for relationship in relationships:
    #     print(relationship[0], relationship[1], relationship[2])