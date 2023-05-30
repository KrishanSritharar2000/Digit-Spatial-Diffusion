# Class to take in 84x84 image representing a 3x3 grid of MNIST digits and finds 
# all the spatial relationships between them. 
# The relationships include "left of", "right of", "above", "below",
from collections import defaultdict
import PIL
from PIL import Image
import numpy as np
from mnist_classifier import MNISTClassifier, CustomTensorDataset
import re
import os
import pickle
import json
from tqdm import tqdm

class Evaluate:
    def __init__(self):
        # Load the classifier
        self.classifier = MNISTClassifier()
        self.classifier.load()
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
        p_digits = list(map(int, re.findall(self.digit_regex, prompt)))
        p_relationships = re.findall(self.relationship_regex, prompt)
        p_relationships_and_digits = [(p_digits[i], p_relationships[i], p_digits[i + 1]) for i in range(len(p_relationships))]
        accuracy = 0
        if len(digits) != len(p_digits):
            return 0
        # for relationship in p_relationships_and_digits:
        #     if relationship in relationships:
        #         accuracy += 1

        prompt_set = set(p_relationships_and_digits)
        image_set = set(relationships)

        matches = prompt_set.intersection(image_set)
        score = len(matches) / len(prompt_set)
        return score
    
    def compute_accurary(self, data):
        accuracy = 0
        for prompt_with_idx in tqdm(data.keys(), desc="Computing accuracy"):
            t_accuracy = 0
            prompt = prompt_with_idx.split("_")[1].strip()
            
            p_digits = list(map(int, re.findall(self.digit_regex, prompt)))
            p_relationships = re.findall(self.relationship_regex, prompt)
            p_relationships_and_digits = [(p_digits[i], p_relationships[i], p_digits[i + 1]) for i in range(len(p_relationships))]

            for img in tqdm(data[prompt_with_idx].keys(), desc="Processing image data", leave=False):
                relationships, digits = data[prompt_with_idx][img]["relationships"], data[prompt_with_idx][img]["digits"]
            
                # If the number of digits is not the same, then the accuracy is 0
                if len(digits) != len(p_digits):
                    t_accuracy += 0
                    continue
                # If a digit is not in the prompt, then the accuracy is 0
                if (set(digits) != set(p_digits)):
                    t_accuracy += 0
                    continue

                prompt_set = set(p_relationships_and_digits)
                image_set = set(relationships)

                # Accuracy is how many relationships are correct (either 0.5 or 1 since there are only 2 in p_relationships)
                matches = prompt_set.intersection(image_set)
                score = len(matches) / len(prompt_set)
                t_accuracy += score
            # print(f"Accuracy for {prompt_with_idx}: {t_accuracy / len(data[prompt_with_idx])}")
            accuracy += t_accuracy / len(data[prompt_with_idx])
        print(f"Total accuracy: {accuracy / len(data)}")
        return accuracy / len(data)


    
    def calculate_relationships_on_testset(self, test_dir):
        # Check that test_dir exists
        if not os.path.exists(test_dir):
            raise ValueError("Test directory does not exist")

        data = defaultdict(dict)        
        #Iterate through the directories in test_dir
        subdirs = os.listdir(test_dir)
        for subdir in tqdm(subdirs, desc="Processing directories"):
            # Check that subdir is a directory
            subdir = os.path.join(test_dir, subdir)
            if not os.path.isdir(subdir):
                continue


            # Get the directory name
            subdir_name = os.path.basename(subdir)
            prompt_name = subdir_name.split('_')[1]
            data[subdir_name] = {}
            # Iterate through the files in the directory
            files = os.listdir(subdir)
            for idx, file in tqdm(enumerate(files), desc=f"Processing {subdir_name}", leave=False):
                #Check filename begins with 'grid'
                if file.startswith('grid'):
                    continue

                #Check the image dimensions
                file_path = os.path.join(subdir, file)
                image = Image.open(file_path)
                file_name = file.split('.')[0]
                
                # Check the size is 128x128
                if image.size != (128, 128):
                    raise ValueError(f"Image {file_path} is not 128x128")

                # Resize the image to 84x84
                image = image.resize((84, 84), resample=PIL.Image.BICUBIC)

                # Find the relationships
                relationships, digits = self.find_relationships(image)

                data[subdir_name][file_name] = {
                    "relationships": relationships,
                    "digits": digits,
                }
        with open('testset_m6e30_3_baseline.pkl', 'wb') as f:
            pickle.dump(data, f)
        # with open('testset_m15e181_baseline.json', 'wb') as f:
        #     json.dumps(data, f)


if __name__ == "__main__":
    e = Evaluate()
    # e.calculate_relationships_on_testset('../stable-diffusion/ldm_test_outputs/test_set_baseline_m6e30_3')
    # print("testset_m6e30_baseline")
    # data1 = pickle.load(open('testset_m6e30_1_baseline.pkl', 'rb'))
    # score1 = e.compute_accurary(data1)
    # data2 = pickle.load(open('testset_m6e30_2_baseline.pkl', 'rb'))
    # score2 = e.compute_accurary(data2)
    # data3 = pickle.load(open('testset_m6e30_3_baseline.pkl', 'rb'))
    # score3 = e.compute_accurary(data3)
    # print("Avg: ", (score1 + score2 + score3)/3)

    # print("testset_m12e152_baseline")
    # data1 = pickle.load(open('testset_m12e152_1_baseline.pkl', 'rb'))
    # score1 = e.compute_accurary(data1)
    # data2 = pickle.load(open('testset_m12e152_2_baseline.pkl', 'rb'))
    # score2 = e.compute_accurary(data2)
    # data3 = pickle.load(open('testset_m12e152_3_baseline.pkl', 'rb'))
    # score3 = e.compute_accurary(data3)
    # print("Avg: ", (score1 + score2 + score3)/3)


    # print("testset_m14e244_baseline")
    # data1 = pickle.load(open('testset_m14e244_1_baseline.pkl', 'rb'))
    # score1 = e.compute_accurary(data1)
    # data2 = pickle.load(open('testset_m14e244_2_baseline.pkl', 'rb'))
    # score2 = e.compute_accurary(data2)
    # data3 = pickle.load(open('testset_m14e244_3_baseline.pkl', 'rb'))
    # score3 = e.compute_accurary(data3)
    # print("Avg: ", (score1 + score2 + score3)/3)


    # print("testset_m15e181_baseline")
    # data1 = pickle.load(open('testset_m15e181_1_baseline.pkl', 'rb'))
    # score1 = e.compute_accurary(data1)
    # data2 = pickle.load(open('testset_m15e181_2_baseline.pkl', 'rb'))
    # score2 = e.compute_accurary(data2)
    # data3 = pickle.load(open('testset_m15e181_3_baseline.pkl', 'rb'))
    # score3 = e.compute_accurary(data3)
    # print("Avg: ", (score1 + score2 + score3)/3)
    # print(len(data))

    
    # Load all the images from the './samples' directory
    # directory = './samples'
    # image_size =  84 # 28*3
    # finder = Evaluate()

    # accuracy = 0
    # samples = 0
    # # Iterate over all files in the directory
    # for filename in os.listdir(directory):
    #     if filename.endswith(".jpg") or filename.endswith(".png"):
    #         # Construct the full file path
    #         file_path = os.path.join(directory, filename)
    #         print(f"Processing {file_path}")

    #         # Get the filename
    #         prompt = "8 right of 4 below 0"
    #         print(prompt)

    #         # Open the image file
    #         image = Image.open(file_path)
    #         image = image.resize((image_size, image_size), resample=PIL.Image.BICUBIC)

            
    #         a = finder.compute_accuracy(prompt, image)
    #         print(f"Accuracy: {a}")
    #         accuracy += a
    #         samples += 1

    #         # Close the image file
    #         image.close()

    # print(f"Average accuracy: {accuracy/samples}")
    