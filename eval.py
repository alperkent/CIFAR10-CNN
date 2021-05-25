import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# unpickles CIFAR-10 data as instructed in README file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# creates a dataset class suitable for loaders 
class Dataset(torch.utils.data.Dataset):
  def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        X = self.transform(self.data[index])
        y = self.labels[index] 
        return X, y

# creates the model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*4*4, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (N, 3, 32, 32)  -> (N, 32, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (N, 32, 16, 16) -> (N, 64, 8, 8)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (N, 64, 8, 8)   -> (N, 128, 4, 4)
        x = torch.flatten(x, 1)                         # (N, 128, 4, 4)  -> (N, 2048)
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc1(x)))  # (N, 2048) -> (N, 512)
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))  # (N, 512)  -> (N, 128)
        x = self.fc3(x)                    # (N, 128)  -> (N, 10)
        return x

    def features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (N, 3, 32, 32)  -> (N, 32, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (N, 32, 16, 16) -> (N, 64, 8, 8)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (N, 64, 8, 8)   -> (N, 128, 4, 4)
        x = torch.flatten(x, 1)                         # (N, 128, 4, 4)  -> (N, 2048)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATH = 'model.py'
batch_size = 64

# normalization
means = [0.491, 0.482, 0.446]
stds = [0.247, 0.244, 0.262]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds)])

# loads test data
test = unpickle('cifar10_data/cifar10_data/test_batch')
testset = test[b'data'].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
testlabels = torch.tensor(test[b'labels'], dtype=torch.long, device=device)
test_set = Dataset(testset, testlabels, transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
del test

# reloads best model
network = CNN()
network.to(device)
network.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = network(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Test accuracy: %.4f %%' % (100*correct/total))