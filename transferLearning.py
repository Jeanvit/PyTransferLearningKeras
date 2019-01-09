from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os, os.path 

#include_top: whether to include the fully-connected layer at the top of the network.
#For transfer learning, just the dense layer will learn about the new classes. 
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Train and test directories
train_dir = '/data2/Train/'
test_dir = '/data2/Test/'

#Training parameters
batch_size = 2
training_epochs = 20

'''
Keras provide an easy way to load an image and generate batches of tensor image data with real-time data augmentation.
The data will be looped over (in batches). This function does exactly that, being as generic as possible to be used both
on train and test sets
'''
def data_generator(network, generator, number_of_images, batch_size, shape_features, shape_labels):
	'''
	inputs:
		network: The network to process the images
		generator: The image_data generator of the given inputs (train or test)
		number_of_images: Number of images present in the inputs
		batch_size: Size of the batch 
		shape_features: Format of the features
		shape_labels: Format of the labels

	return:
		features and labels of the given class after a single pass through the network 
        of all input images
	'''
	features = np.zeros(shape=shape_features)
	labels = np.zeros(shape=shape_labels)
	i = 0
	for inputs_batch, labels_batch in train_generator:
        
        #Due to the network format, every image goes through it to have the expected dimensions
		features_batch = network.predict(inputs_batch)
		features[i * batch_size : (i + 1) * batch_size] = features_batch
		labels[i * batch_size : (i + 1) * batch_size] = labels_batch
		i += 1
		if i * batch_size >= number_of_images:
			break
	return features, labels

#General use function to count the number of images and folders (classes) inside a directory
def count_number_of_folders_and_files(directory):
	'''
	inputs:
		directory: The directory where the recursive process will take place

	return:
		totalFiles,totalFolders The number of files and folders
	'''
	totalFolders = 0
	totalFiles = 0
	for root, dirs, files in os.walk(directory):
		totalFiles += len(files)
		totalFolders += len(dirs)
	return  totalFiles,totalFolders


#Load the image data
def imagedata_generator(folder, size, batch_size):
	'''
		inputs:
			folder: the directory where the images are located
			size: The size of the image
			batch_size: The size of the batch 
		return:
			An ImageDataGenerator containing all the images of a given directory
	'''
	imagedata_gen = ImageDataGenerator(rescale=1./255)
	imagedata_gen = imagedata_gen.flow_from_directory(folder, target_size=size, batch_size=batch_size, class_mode='categorical', shuffle=True)
	return imagedata_gen

print ("Counting folders and files...")
number_of_training_images, number_of_classes = count_number_of_folders_and_files(train_dir)
number_of_testing_images, number_of_classes =  count_number_of_folders_and_files(test_dir)
print ("Done!")


print ("Train data:")
train_generator = imagedata_generator(folder=train_dir, size=(224, 224), batch_size=batch_size)

print ("Test data:")
test_generator = imagedata_generator(folder=test_dir, size=(224, 224), batch_size=batch_size)

print ("Generating train features and labels")
train_features, train_labels = data_generator(vgg_conv, train_generator, number_of_training_images, batch_size, 
	shape_features=(number_of_training_images, 7, 7, 512), shape_labels=(number_of_training_images,number_of_classes))

print ("Generating test features and labels")
test_features, test_labels = data_generator(vgg_conv, test_generator, number_of_testing_images, batch_size, 
	shape_features=(number_of_testing_images, 7, 7, 512), shape_labels=(number_of_testing_images,number_of_classes))
print ("Done!")

train_features = np.reshape(train_features, (number_of_training_images, 7 * 7 * 512))
test_features = np.reshape(test_features, (number_of_testing_images, 7 * 7 * 512))


from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(number_of_classes, activation='softmax'))

#Before training a model, you need to configure the learning process, which is done via the compile method.
model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_features,
                    train_labels,
                    epochs=training_epochs,
                    batch_size=batch_size, verbose=2, # show the progress one time per epoch
                    validation_data=(test_features,test_labels))


import matplotlib.pyplot as plt
%matplotlib inline

#Model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()