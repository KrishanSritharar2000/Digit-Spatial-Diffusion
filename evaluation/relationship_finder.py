# Class to take in 84x84 image representing a 3x3 grid of MNIST digits and finds 
# all the spatial relationships between them. 
# The relationships include "left of", "right of", "above", "below",
import PIL
from PIL import Image
import numpy as np
from mnist_classifier import MNISTClassifier


class DigitRelationshipFinder:
    def __init__(self, classifier):
        self.classifier = classifier

    def find_relationships(self, image):
        # Convert PIL Image to numpy array
        image = np.array(image)

        # Split the image into 9 digits
        digits = [Image.fromarray(image[28*(i//3):28*((i//3)+1), 28*(i%3):28*((i%3)+1)]) for i in range(9)]


        # Identify each digit using the classifier
        identified_digits = [self.classifier.classify_digit(digit) for digit in digits]

        relationships = []

        # Find the relationships
        for i, digit in enumerate(identified_digits):
            # Compute the position in the 3x3 grid
            row, col = divmod(i, 3)

            # Check for a digit to the right
            if col < 2:
                relationships.append((digit, "right of", identified_digits[i+1]))

            # Check for a digit to the left
            if col > 0:
                relationships.append((digit, "left of", identified_digits[i-1]))

            # Check for a digit below
            if row < 2:
                relationships.append((digit, "below", identified_digits[i+3]))

            # Check for a digit above
            if row > 0:
                relationships.append((digit, "above", identified_digits[i-3]))

        return relationships

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