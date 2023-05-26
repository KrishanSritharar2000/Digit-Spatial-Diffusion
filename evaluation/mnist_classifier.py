import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, ConcatDataset
import pickle

# Check if GPU is available and if not, fall back on CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the architecture of the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 11)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class MNISTClassifier:

    def __init__(self):
        self.load_data()

    def load_data(self):
        # Assuming no_digit_dataset is your custom dataset
        no_digit_dataset_train = torch.load('no_digit_dataset_train.pth')
        no_digit_dataset_test = torch.load('no_digit_dataset_test.pth')

        # Transform the data to torch tensors and normalize it 
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), lambda x: x.long()])

        # Prepare training set and testing set
        trainset = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)

        # Combine the datasets
        train_dataset = ConcatDataset([trainset, no_digit_dataset_train])
        test_dataset = ConcatDataset([testset, no_digit_dataset_test])
        # train_dataset = trainset
        # test_dataset = testset

        # Prepare data loaders
        self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        

    def print_sample_types(self, dataset, num_samples=5):
        for i in range(num_samples):
            image, label = dataset[i]
            print(f"Sample {i}:")
            print("Image type: ", type(image))
            print("Label type: ", type(label))

        # print("MNIST training set:")
        # print_sample_types(trainset)
        # print("\nNo digit training set:")
        # print_sample_types(no_digit_dataset_train)

    def train(self):
        # Instantiate the network, the loss function and the optimizer
        self.net = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        # Train the network showing the loss and progress using tqdm
        for epoch in tqdm(range(10)):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in tqdm(enumerate(self.trainloader, 0)):
                # get the inputs
                inputs, labels = data[0].float().to(device), data[1].to(device)


                # zero the parameter gradients
                optimizer.zero_grad()
                # print("inputs type: ", type(inputs), inputs.dtype)
                # print("labels type: ", type(labels), labels.dtype)


                # forward propagation
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                
                # backward propagation
                loss.backward()
                
                # optimize
                optimizer.step()
                
                # print statistics
                running_loss += loss.item()
                
            print('[%d] loss: %.3f' % (epoch + 1, running_loss / 2000))
            running_loss = 0.0


        print('Finished Training')

        # Save the model
        PATH = './mnist_classifier_no_digit_inc.pth'
        torch.save(self.net.state_dict(), PATH)

    def test(self):
        # Test the network
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].float().to(device), data[1].to(device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the {round(len(self.testloader)*64, 0)} test images: %d %%' % (100 * correct / total))

    def load(self):
        # Load the model
        self.net = Net()
        # PATH = './mnist_classifier.pth'
        PATH = './mnist_classifier_no_digit_inc.pth'
        self.net.load_state_dict(torch.load(PATH))
        self.net.to(device)

    def classify_digit(self, digitImage):
        #Convert the image to a tensor
        digitTensor = transforms.ToTensor()(digitImage)
        digitTensor = digitTensor.to(device)
        # Classify a digit
        with torch.no_grad():
            output = self.net(digitTensor)
            _, predicted = torch.max(output.data, 1)
            return predicted.item()
        
    def create_no_digit_dataset(self):
        no_digit_images_zero = torch.zeros([3000, 1, 28, 28])
        no_digit_images_rand = 2 * torch.rand([3000, 1, 28, 28]) - 1
        no_digit_images = torch.cat((no_digit_images_zero, no_digit_images_rand), 0)
        #shuffle the images
        no_digit_images = no_digit_images[torch.randperm(no_digit_images.size()[0])]
        no_digit_labels = torch.full((6000,), 10)
        no_digit_images = no_digit_images.to(torch.float32)
        no_digit_labels = no_digit_labels.to(torch.int64)
        self.no_digit_dataset = CustomTensorDataset(no_digit_images, no_digit_labels)
        # self.no_digit_dataset = TensorDataset(no_digit_images, no_digit_labels)
        with open('./no_digit_dataset_train.pkl', 'wb') as f:
            pickle.dump(self.no_digit_dataset, f)
        torch.save(self.no_digit_dataset, './no_digit_dataset_train.pth')

class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx].item()


if __name__ == "__main__":
    mnist_classifier = MNISTClassifier()
    # mnist_classifier.train()
    mnist_classifier.load()
    mnist_classifier.test()
    # mnist_classifier.create_no_digit_dataset()