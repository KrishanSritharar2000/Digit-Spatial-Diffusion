import numpy as np
from sklearn.model_selection import train_test_split
import pickle

num_images = 10000
indices = np.arange(num_images)

# Split the indices into train and test sets
train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=3168)

# Save the indices to pickle files
with open('train_indices.pkl', 'wb') as f:
    pickle.dump(train_indices, f)
with open('test_indices.pkl', 'wb') as f:
    pickle.dump(test_indices, f)

# with open('test_indices.pkl', 'rb') as f:
#     test_indices = pickle.load(f)
#     print(test_indices)
# with open('train_indices.pkl', 'rb') as f:
#     train_indices = pickle.load(f)
#     print(train_indices)