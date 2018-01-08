
# coding: utf-8

# # *** Transfer learning using InseptionResNetV2 ***

# Imports

# In[ ]:

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


# Initialisng parameters

# In[ ]:

img_width, img_height = 139, 139 # the image dimensions supported by InceptionResNetV2 and InceptionV3, this saves space on GPU
train_data_dir = "data/train" # the training directory 
validation_data_dir = "data/test" # the directory we will use for validation


# Setting the model

# In[ ]:

#model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))
model = InceptionV3(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))
# we maxed out the resolution for the model to gain the most information
# we put a 3 indicating we are taking in the imaages in RGB(colour)


# In[ ]:

model.summary() # seeing the sctructure of the neural network


# In[ ]:

## if you have layers you don't want you can ignore them
# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False


# In[ ]:

##model.summary() # seeing the sctructure of the neural network


# ## ___Adding layers onto the pre-existing architecture___

# In[ ]:

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(32, activation="relu")(x)


# Defining the fully connected layer

# In[ ]:

predictions = Dense(units = 1, activation="sigmoid")(x) # sigmoid is for yes or no training oprions


# Defining the model for training

# In[ ]:

# creating the final model for computation
model_final = Model(inputs = model.input, outputs = predictions)


# Compiling the model

# In[ ]:

# compile the model 
model_final.compile(loss = 'binary_crossentropy', optimizer = "Adamax", metrics=["accuracy"])
#Adamax is a variant of Adam optimizer but l prefer the better learning rate it has and...
# it is lighter for computation in comparison with Adadelta


# ## ___Data preparation___

# In[ ]:

# Initiate the training data generator with data Augumentation 
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)


# In[ ]:

# Initiate the test data generator with data Augumentation 
test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)


# ## ___Initialising the model for training___

# In[ ]:

## number of images to take as batches to train
batch_size = 1 #one image is easier is process allowing resources to be better utilized at thhe expense of time though


# In[ ]:

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "binary")


# In[ ]:

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "binary")


# ## ___Parameters for saving the model___

# In[ ]:

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("cat_dog_model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# ## ___Training the model___

# In[ ]:

# the number of epochs we want to train our data 
epochs = 5


# In[ ]:

# Train the model

# steps_per_epoch - It should typically be equal to the number of samples of your dataset divided by the batch size. 
# validation_steps - It should typically be equal to the number of samples of your validation dataset divided by the batch size. 
model_final.fit_generator(
train_generator,
steps_per_epoch = 1000,
epochs = epochs,
verbose=1,
validation_data = validation_generator,
callbacks = [checkpoint, early],
validation_steps = 250)


# In[ ]:

print ('Done training and saving model \n Thank you boss')


# _____ Done _____
