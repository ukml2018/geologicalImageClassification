from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image


SRCDIR = os.path.dirname(os.path.abspath(__file__))
print("The Source Directory :", SRCDIR)
PATH = os.path.join(SRCDIR, 'images')
print("The Data directory: ", PATH)
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

#-- training data set
train_andesite_dir = os.path.join( train_dir, 'andesite')
train_gneiss_dir = os.path.join( train_dir, 'gneiss')
train_marble_dir = os.path.join( train_dir, 'marble')
train_quartzite_dir = os.path.join( train_dir, 'quartzite')
train_rhyolite_dir = os.path.join( train_dir, 'rhyolite')
train_schist_dir = os.path.join( train_dir, 'schist')

#-- validation data set
validation_andesite_dir = os.path.join( validation_dir, 'andesite')
validation_gneiss_dir = os.path.join( validation_dir, 'gneiss')
validation_marble_dir = os.path.join( validation_dir, 'marble')
validation_quartzite_dir = os.path.join( validation_dir, 'quartzite')
validation_rhyolite_dir = os.path.join( validation_dir, 'rhyolite')
validation_schist_dir = os.path.join( validation_dir, 'schist')

#-- Count number of training and validation data
num_andesite_tr = len(os.listdir(train_andesite_dir))
num_gneiss_tr = len(os.listdir(train_gneiss_dir))
num_marble_tr = len(os.listdir(train_marble_dir))
num_quartzite_tr = len(os.listdir(train_quartzite_dir))
num_rhyolite_tr = len(os.listdir(train_rhyolite_dir))
num_schist_tr = len(os.listdir(train_schist_dir))

num_andesite_val = len(os.listdir(validation_andesite_dir))
num_gneiss_val = len(os.listdir(validation_gneiss_dir))
num_marble_val = len(os.listdir(validation_marble_dir))
num_quartzite_val = len(os.listdir(validation_quartzite_dir))
num_rhyolite_val = len(os.listdir(validation_rhyolite_dir))
num_schist_val = len(os.listdir(validation_schist_dir))

total_train = num_andesite_tr + num_gneiss_tr + num_marble_tr + num_quartzite_tr + num_rhyolite_tr + num_schist_tr
total_val = num_andesite_val + num_gneiss_val + num_marble_val + num_rhyolite_val + num_schist_val

print(' total training andesite images: ', num_andesite_tr)
print(' total training gneiss images: ', num_gneiss_tr)
print(' total training marble images: ', num_marble_tr)
print(' total training quartzite images: ', num_quartzite_tr)
print(' total training rhyolite images: ', num_rhyolite_tr)
print(' total training schist images: ', num_schist_tr)

print(' total validation andesite images: ', num_andesite_val)
print(' total validation gneiss images: ', num_gneiss_val)
print(' total validation marble images: ', num_marble_val)
print(' total validation quartzite images: ', num_quartzite_val)
print(' total validation rhyolite images: ', num_rhyolite_val)
print(' total validation schist images: ', num_schist_val)

print(" -------- ")
print('Total training images:',total_train )
print('Total validation images:',total_val )

#-- Set up variables
batch_size = 700
epochs = 15
IMG_HEIGHT = 200
IMG_WIDTH = 200

#-- Prepare the data
train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range= 45,
                                           width_shift_range=0.15,
                                           height_shift_range=0.15,
                                           horizontal_flip=True,
                                           zoom_range=0.5)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                              class_mode='binary')



#-- Visualize training Images
sample_training_images, _ = next(train_data_gen)

#This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column
def plotImages(image_arr):
    fig, axes = plt.subplots(1,5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip (image_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Zoom augmentation
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
#plotImages(sample_training_images[:5])

#-- Create the Model
model = Sequential([
       Conv2D(16,3, padding='same' , activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
       MaxPooling2D(),
       Dropout(0.3),
       Conv2D(32,3, padding='same' , activation='relu'),
       MaxPooling2D(),
       Conv2D(64,3, padding='same' , activation='relu'),
       MaxPooling2D(),
       Dropout(0.3),
       Flatten(),
       Dense(512,activation='relu'),
       Dropout(0.1),
       Dense(1,activation='sigmoid')
       ])

#--Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#-- Model Summary
model.summary()

#-- Train the Model
history = model.fit_generator(
          train_data_gen,
          steps_per_epoch=total_train // batch_size,
          epochs= epochs,
          validation_data=val_data_gen,
          validation_steps=total_val // batch_size
)

#Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label= 'Training Accuracy')
plt.plot(epochs_range, val_acc, label= 'Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label= 'Validation Loss')
plt.plot(epochs_range, val_loss, label= 'Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()







