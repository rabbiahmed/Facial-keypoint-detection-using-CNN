# Facial Keypoint Detection

## Project Overview

In this project, a facial keypoint detection system was built using computer vision techniques and deep neural network architectures. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. The developed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face.

The project will be broken up into the following Python notebooks:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/P1_Facial_Keypoints.git
cd P1_Facial_Keypoints
```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```
	
	At this point your command line should look something like: `(cv-nd) <User>:P1_Facial_Keypoints <user>$`. The `(cv-nd)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

### Data

All of the data that were needed to train a neural network is in the Facial_Keypoints repo, in the subdirectory `data`. In this folder, there are training and tests set of image/keypoint data, and their respective csv files. 

## Methods

### Notebooks

1. Navigate back to the repo. (Also, the source environment should still be activated at this point.)
```shell
cd
cd Facial_Keypoints
```

2. Open the directory of notebooks, using the below command. Verify that all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

3. Once you open any of the project notebooks, make sure you are in the correct `cv-nd` environment by clicking `Kernel > Change Kernel > cv-nd`.

### `models.py`

#### Specify the CNN architecture
|  Define a CNN in `models.py`. |  Define a convolutional neural network with at least one convolutional layer, i.e. self.conv1 = nn.Conv2d(1, 32, 5). The network takes in a grayscale, square image. |

### Notebook 2

#### Define the data transform for training and test data
|  Define a `data_transform` and applied it whenever we instantiate a DataLoader. |  The composed transform includes: rescaling/cropping, normalization, and turning input images into torch Tensors. The transform turns any input image into a normalized, square, grayscale image and then a Tensor for the model to take it as input. |

#### Define the loss and optimization functions
|  Select a loss function and optimizer for training the model. |  The loss and optimization functions should be appropriate for keypoint detection, which is a regression problem. |

#### Train the CNN
| Train the model.  |  Train the CNN after defining its loss and optimization functions. It is a good idea to visualize the loss over time/epochs by printing it out occasionally and/or plotting the loss over time. Then, save the best trained model. |

#### Visualize one or more learned feature maps
| Apply a learned convolutional kernel to an image and investigate its effects. |  The CNN "learns" (updates the weights in its convolutional layers) to recognize features and this step requires that we extract at least one convolutional filter from the trained model, apply it to an image, and analyze further what effect this filter has on the image. |

### Notebook 3

#### Detect faces in a given image
| Use a haar cascade face detector to detect faces in a given image. | 

#### Transform each detected face into an input Tensor
| Turn each detected image of a face into an appropriate input Tensor. | We should transform any face into a normalized, square, grayscale image and then a Tensor for the model to take in as input (similar to what the `data_transform` did in Notebook 2). |

#### Predict and display the keypoints
| Predict and display the keypoints on each detected face. | After face detection with a Haar cascade and face pre-processing, apply the trained model to each detected face, and display the predicted keypoints on each face in the image. |

This project was completed as a part of the Udacity Computer Vision course.

LICENSE: This project is licensed under the terms of the MIT license.
