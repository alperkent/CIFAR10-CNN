"""
Next 4 cells are for installing a t-SNE package (tsne-cuda) that can utilize GPU to handle much faster calculations.

tsne-cuda: https://github.com/CannyLab/tsne-cuda
"""

# IPython code to install conda
# %%bash
# MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
# MINICONDA_PREFIX=/usr/local
# wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
# chmod +x $MINICONDA_INSTALLER_SCRIPT
# ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
# conda install --channel defaults conda python=3.7 --yes
# conda update --channel defaults --all --yes

# IPython code to install tsne-cuda and necessary packages
# !yes Y | conda install faiss-gpu cudatoolkit=10.1 -c pytorch
# !apt search openblas
# !yes Y | apt install libopenblas-dev
# !wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda101.tar.bz2
# !tar xvjf tsnecuda-2.1.0-cuda101.tar.bz2 
# !cp -r site-packages/* /usr/local/lib/python3.7/dist-packages/
# !echo $LD_LIBRARY_PATH 
# !ln -s /content/lib/libfaiss.so $LD_LIBRARY_PATH/libfaiss.so

# This code does a t-SNE on 5000 points, so it should complete relatively quickly (1-2 seconds). If there are no error messages and it doesn't hang, you should be good to go.
# import tsnecuda
# tsnecuda.test()

"""
Actual code starts here:
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from tsnecuda import TSNE
from sklearn.manifold import TSNE

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

# creates a TSNE model and plots it
def plot_TSNE(network, loader, epoch):
    data = []
    targets = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = network.features(inputs).cpu().numpy()
            data.append(outputs)
            targets.append(labels.cpu().numpy())
    data = np.array(data).reshape((-1, 2048))
    targets = np.array(targets).reshape((-1))
    tsne_data = TSNE(perplexity=50, n_iter=1000).fit_transform(data)
    for i in range(len(classes)):
      plt.scatter(tsne_data[np.where(targets[:] == i), 0], tsne_data[np.where(targets[:] == i), 1], s=0.5, label=classes[i])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Latent space at epoch %d' % (epoch+1))
    plt.show()

# uses GPU to train the model determinisitically for reproducibility
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(421)
# np.random.seed(421)
# torch.use_deterministic_algorithms(True)

# loads training data 
train1 = unpickle('cifar10_data/cifar10_data/data_batch_1')
train2 = unpickle('cifar10_data/cifar10_data/data_batch_2')
train3 = unpickle('cifar10_data/cifar10_data/data_batch_3')
train4 = unpickle('cifar10_data/cifar10_data/data_batch_4')
trainset = np.concatenate((train1[b'data'], train2[b'data'], train3[b'data'], train4[b'data'])).reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
trainlabels = torch.tensor(np.concatenate((train1[b'labels'], train2[b'labels'], train3[b'labels'], train4[b'labels'])), dtype=torch.long, device=device)
del train1, train2, train3, train4

# loads validation data
train5 = unpickle('cifar10_data/cifar10_data/data_batch_5')
valset = train5[b'data'].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
vallabels = torch.tensor(train5[b'labels'], dtype=torch.long, device=device)
del train5

# loads class labels
label_info = unpickle('cifar10_data/cifar10_data/batches.meta')
classes = [x.decode('utf-8') for x in label_info[b'label_names']] 
del label_info

# hyperparameters
batch_size = 64
max_epochs = 100
learning_rate = 0.001

# transformations
means = [trainset[:, :, :, 0].mean()/255, 
         trainset[:, :, :, 1].mean()/255, 
         trainset[:, :, :, 2].mean()/255]
stds = [trainset[:, :, :, 0].std()/255, 
        trainset[:, :, :, 1].std()/255, 
        trainset[:, :, :, 2].std()/255]
transform_train = transforms.Compose([
    transforms.ToPILImage(),                                  
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds)])

# loaders
training_set = Dataset(trainset, trainlabels, transform_train)
trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

validation_set = Dataset(valset, vallabels, transform)
valloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)

# functions to show an image
def imshow(image):
    image_np = np.transpose(image.numpy(), (1, 2, 0))
    image_np = image_np * stds + means
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))

# initializes the network, loss function and optimizer
network = CNN()
network.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adagrad(network.parameters(), lr=learning_rate)
# optimizer = optim.RMSprop(network.parameters(), lr=learning_rate)

# initializes early stopping parameters
stop = {'patience': 5, 'wait': 0, 'best_error': 1, 'best_epoch': 0}

# path to file for saving model parameters
PATH = 'model.py'

# loops over epochs
losses = []
train_acc = []
val_acc = []
for epoch in range(max_epochs):
    # training
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        running_loss += loss.item()
    train_acc.append(100*train_correct/train_total)
    losses.append(running_loss/train_total)
        
    # validation
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc.append(100*val_correct/val_total)

    print('Loss at epoch %18d: %f' % (epoch+1, losses[-1]))
    print('Training accuracy at epoch %5d: %.4f %%' % (epoch+1, train_acc[-1]))
    print('Validation accuracy at epoch %3d: %.4f %%' % (epoch+1, val_acc[-1]))
    print('-------------------------------------------')

    # plots t-SNE at the beginning, middle, and end of training
    # if epoch == 0 or epoch == 13 or epoch == 28:
    #     plot_TSNE(network, trainloader, epoch)
    #     print('-------------------------------------------')

    # evaluates early stopping and saves the model parameters accordingly
    stop['current_error'] = 1 - val_acc[-1] / 100
    if stop['current_error'] < stop['best_error']:
        stop['best_error'] = stop['current_error']
        stop['best_epoch'] = epoch
        torch.save(network.state_dict(), PATH)
        stop['wait'] = 1
    else:
        if stop['wait'] >= stop['patience']:
            print('Terminated training for early stopping at epoch %d' % (epoch+1))
            break
        stop['wait'] += 1

print('Final epoch for best model: %d' % (stop['best_epoch']+1))
print('Final training accuracy: %.4f %%' % train_acc[stop['best_epoch']])
print('Final validation accuracy: %.4f %%' % val_acc[stop['best_epoch']])

# plots costs over epochs
plt.plot(range(1, stop['best_epoch']+2), losses[:stop['best_epoch']+1])
plt.title('Cost per epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

# plots training and validation errors over epochs
plt.plot(range(1, stop['best_epoch']+2), [1-x/100 for x in train_acc[:stop['best_epoch']+1]], label='Training')
plt.plot(range(1, stop['best_epoch']+2), [1-x/100 for x in val_acc[:stop['best_epoch']+1]], label='Validation')
plt.title('Error per epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.show()