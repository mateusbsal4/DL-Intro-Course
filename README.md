# Introduction to Deep Learning - Tutorials

This repository contains the code for the Introduction to Deep Learning lecture at TUM. The exercises primarily focus on Computer Vision tasks, exploring various neural network architectures ranging from the well known fully connected networks to autoencoders, CNNs and fully convolutional networks.

In the first four exercises below, Numpy was used to implement the core components of a deep learning pipeline in a more comprehensive manner.

## Exercise 3 - Dataset and Dataloader
Numpy implementation of the Dataset class - properly reads the input data, performs eventual preprocessing and augmentations - and the Dataloader classes

## Exercise 4 - Simple Classifier
Numpy implementation of a simple model and Solver classes - contwining the training loop (forward and backward passes), optimizer method - which performs a SGD based optimization and the validation of the pipeline

## Exercise 5 - Neural Networkds
Numpy implementation as above of the loss function and Solver class (forward and backward passes for the affine layers, ) for a multi-class classification task with the CIFAR-10 dataset, with a predefined network architecture and hyper-parameters


## Exercise 6 - Hyperparameter tuning
The goal of this exercise was to select and tune the architecture and hyperparameters of the fully connected NN of the above task to surpass an accuracy threshold on the test set.



## Exercise 8 - Autoencoder
The goal of the exercise was to use transfer learning to surpass an accuracy threshold for the classification of digits using a mostly unlabeled MNIST dataset (57700 unlabeled and 300 labeled images). The implementation and training of the encoder, decoder and classifier models was done in PyTorch.

Below are the reconstructed validation images from the autoencoder:

![Screenshot from 2024-04-13 13-04-44](https://github.com/mateusbsal4/DL-Intro-Course/assets/84996618/696e3923-c92f-4cf2-95f9-598cffdac5c4)

## Exercise 7 - Intro to PyTorch and Tensorboard
The goal of the exercise was to successfully implement a a fully connected NN image classifier for CIFAR-10 with PyTorch Lightning

## Exercise 9 - Facial Keypoint Detection
The goal of the exercise was to implement and train a convolution neural network for facial keypoint detection in PyTorch.

Below are the predictions of the model (red dots) and the ground truth keypoints (blue dots):

![Screenshot from 2024-04-13 13-05-35](https://github.com/mateusbsal4/DL-Intro-Course/assets/84996618/1f9a4243-3674-409f-a92f-b941335c5974)

## Exercise 10 - Semantic Segmentation
The goal of this exercise was to implement and train a fully convolutional neural network for Semantic Segmentation.

Below are the predictions of the model compared against the target images (labeled by hand) for the input data:
![Screenshot from 2024-04-13 13-06-25](https://github.com/mateusbsal4/DL-Intro-Course/assets/84996618/8f971f3c-c490-44a8-92a4-8218ae94565f)

