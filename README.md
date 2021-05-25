# CMPE597 Project 2 - CIFAR-10 image classification with CNN

# Description:
This is a Python 3 implementation using Numpy for necessary operations, Matplotlib for visualization, Pickle to load data, Pytorch for creating and training the neural network architecture and saving/loading learned model parameters, and tsne-cuda for t-SNE plots.

# Files:
main.py file loads and augments the dataset, builds the network architecture, trains the model and evaluates it on validation set during training, saves the model parameters to model.py file and outputs cost, training accuracy, validation accuracy, and t-SNE plots and termination epoch & final accuracy values.

eval.py file loads the learned model parameters and evaluates the model on test data, outputting test accuracy.

model.py file contains learned model parameters.

# Instructions:
You can run the model after extracting the data files to "\cifar10data\cifar10data" folders under the same directory with model files. After this step, typing these commands in IPython should return the outputs:

In [1]: %cd "CURRENT DIRECTORY"

In [2]: %run main.py

In [3]: %run eval.py

WARNING: I have commented (#) the code for t-SNE. This is because tsne-cuda package is not supported on all platforms. I originally trained the model on Google Colaboratory and I had to install conda and some other necessary packages to run tsne-cuda. If your operating system is Linux, you can use the commented code in the beginning of the main file on IPython to download necessary packages. Since the tsne-cuda package only works on Linux, you should instead import sklearn package on other operating systems. Beware that training will take much, much longer due to very slow t-SNE training with the sklearn module. 

Google Colaboratory file link: https://colab.research.google.com/drive/1MHQMMUP7p9pQaFbWgcqEDw7p4EPtCegR?usp=sharing