# About

With Transfer Learning is possible to take a pre-trained network (for a set of images for instance), and use it as a starting point for the training of a new task. For example, A network trained for recognizing motorcycles can be retrained to identify bicycles. Note that the problems are different, yet, related to each other. This code is part of a post on [my blog](https://jeanvitor.com/transfer-learning-applied-to-image-recognition-of-similiar-datasets/)


# Requirements

* Python
* Pillow
* Numpy
* Tensorflow
* Keras
* Matplotlib

# Using

This code loads the VGG16 model trained using Imagenet images. To use transfer learning, you must use the following directories with the new dataset.

``` 
train_dir = '/data2/Train/'
test_dir = '/data2/Test/'
```