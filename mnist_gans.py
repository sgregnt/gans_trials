# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import keras
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Activation, BatchNormalization, UpSampling2D, Reshape
from keras.utils import plot_model
from keras import optimizers
from keras import backend as K
## Discriminator

disc_net = Sequential()
input_shape = (28, 28, 1)
prob = 0.4

# filters is the number of channels
l1 = Conv2D(filters = 64, # (the number of output filters in the convolution).
        kernel_size = 5,#specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
        strides = 2,
        input_shape = input_shape,
        padding = 'same') # results in padding the input such that the output has the same length as the original input.)

l2 = LeakyReLU()

l3 = Conv2D(128, 5, strides=2, padding='same') 

l4 = LeakyReLU()

l5 = Dropout(prob)

l6 = Conv2D(256, 5, strides=2, padding='same')

l7 = LeakyReLU()

l8 = Dropout(prob)

l9 = Conv2D(512, 5, strides=1, padding='same')

l10 = LeakyReLU()

l11 = Dropout(prob)

l12 = Flatten()

l13 = Dense(units = 1) #  Positive integer, dimensionality of the output space.

l14 = Activation('sigmoid')

disc_net.add(l1)
disc_net.add(l2)
disc_net.add(l3)
disc_net.add(l4)
disc_net.add(l5)
disc_net.add(l6)
disc_net.add(l7)
disc_net.add(l8)
disc_net.add(l9)
disc_net.add(l10)
disc_net.add(l11)
disc_net.add(l12)
disc_net.add(l13)
disc_net.add(l14)

plot_model(disc_net, to_file='model.png')

# like net_discriminator in the notebook
disc_net.model.summary()


## Generator 

gen_net = Sequential()
prob_g = 0.4


# input is noise vector of length 100
#? normal convolutions instead of transposed comvalutions
# Batch normalization is added to improve stability

g1 = Dense(7 * 7 * 256, input_dim = 100)
g2 = BatchNormalization(momentum = 0.9)
g3 = LeakyReLU()
g4 = Reshape((7,7,256))
g5 = Dropout(prob_g)
g6 = UpSampling2D()
g7 = Conv2D(128, 5, padding = 'same')
g8 = BatchNormalization(momentum = 0.9)
g9 = LeakyReLU()
g10 = UpSampling2D()
g11 = Conv2D(64, 5, padding = 'same')
g12 = BatchNormalization(momentum = 0.9) 
g13 = LeakyReLU()
g14 = Conv2D(32, 5, padding = 'same')
g15 = BatchNormalization(momentum = 0.9)
g16 = LeakyReLU()
g17 = Conv2D(1, 5, padding='same')
g18 = Activation('sigmoid')

gen_net.add(g1)
gen_net.add(g2)
gen_net.add(g3)
gen_net.add(g4)
gen_net.add(g5)
gen_net.add(g6)
gen_net.add(g7)
gen_net.add(g8)
gen_net.add(g9)
gen_net.add(g10)
gen_net.add(g11)
gen_net.add(g12)
gen_net.add(g13)
gen_net.add(g14)
gen_net.add(g15)
gen_net.add(g16)
gen_net.add(g17)
gen_net.add(g18)

def generator():
    
    net = Sequential()
    dropout_prob = 0.4
    
    net.add(Dense(7*7*256, input_dim=100))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    net.add(Reshape((7,7,256)))
    net.add(Dropout(dropout_prob))
    
    net.add(UpSampling2D())
    net.add(Conv2D(128, 5, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    
    net.add(UpSampling2D())
    net.add(Conv2D(64, 5, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    
    net.add(Conv2D(32, 5, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    
    net.add(Conv2D(1, 5, padding='same'))
    net.add(Activation('sigmoid'))
    
    return net

net_generator = generator()
net_generator.summary()
# this is like net_generator in the notebook
#
gen_net = generator()# .model.summary()
gen_net.summary()
# Creating models
# For the discriminator model we only have to define the optimizer, 
# all the other parts of the model are already defined. 
# I have tested both SGD, RMSprop and Adam for the optimizer of 
# the discriminator but RMSprop performed best. RMSprop is used 
# a low learning rate and I clip the values between -1 and 1. 
# A small decay in the learning rate can help with stabilizing.
#  Besides the loss we also tell Keras to gives us the accuracy as a metric.


# optim_disc = optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-10)

# this is like model_discriminator 
# disc_net.compile(loss='binary_crossentropy', optimizer=optim_disc, metrics=['accuracy'])

optim_discriminator = optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-10)
model_discriminator = Sequential()
model_discriminator.add(disc_net)
model_discriminator.compile(loss='binary_crossentropy', optimizer=optim_discriminator, metrics=['accuracy'])

model_discriminator.summary()


#------------------------------------------------------------------------------ 

optim_gen = optimizers.Adam(lr=0.0004, clipvalue=1.0, decay=1e-10)
model_adversarial = Sequential()
model_adversarial.add(gen_net)

# Disable layers in discriminator
for layer in disc_net.layers:
    layer.trainable = False

model_adversarial.add(disc_net)
model_adversarial.compile(loss='binary_crossentropy', optimizer=optim_gen, metrics=['accuracy']) 

model_adversarial.summary()

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 

# reading data
x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)



# Training the GAN
# With our models defined and the data 
# loaded we can start training our GAN. 
# The models are trained one after another, 
# starting with the discriminator. 
# The discriminator is trained on a data set 
# of both fake and real images and tries 
# to classify them correctly. 
# The adversarial model is trained on 
# noise vectors as explained above


import matplotlib.pyplot as plt
from IPython.display import clear_output, Image

batch_size = 256

vis_noise = np.random.uniform(-1.0, 1.0, size=[16, 100])

loss_adv = []
loss_dis = []
acc_adv = []
acc_dis = []
plot_iteration = []
i = 0
for _ in range(50):
    
    print("i", i)
    
    # Select a random set of training images from the mnist dataset
    images_train = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
    # Generate a random noise vector
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    # Use the generator to create fake images from the noise vector
    images_fake = gen_net.predict(noise)
    
    if False: 
        plt.imshow(images_fake[2,:,:,0], cmap='gray')
        plt.show()
        
    # Create a dataset with fake and real images
    x = np.concatenate((images_train, images_fake))
    y = np.ones([2*batch_size, 1])
    y[batch_size:, :] = 0 

    # Train discriminator for one batch
    d_stats = model_discriminator.train_on_batch(x, y)
    
    if  True: #i % 2 == 0:
        # Train the generator
        # The input of the adversarial model is a list of noise vectors. 
        # The generator is 'good' if the discriminator classifies
        # all the generated images as real. 
        # Therefore, the desired output is a list of all ones.
          y = np.ones([batch_size, 1])
          noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
          a_stats = model_adversarial.train_on_batch(noise, y)
        
    if i % 2 == 0:
        plot_iteration.append(i)
        loss_adv.append(a_stats[0])
        loss_dis.append(d_stats[0])
        acc_adv.append(a_stats[1])
        acc_dis.append(d_stats[1])

        clear_output(wait=True)
        
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(16, 8)
        x = list(range(len(loss_adv)))
        ax1.plot(x, loss_adv, label="loss adversarial")
        ax1.plot(x, loss_dis, label="loss discriminator")
        ax1.set_ylim([0,5])
        ax1.legend()

        ax2.plot(x, acc_adv, label="acc adversarial")
        ax2.plot(x, acc_dis, label="acc discriminator")
        ax2.legend()

        plt.show()
        
        plt.imshow(images_fake[5,:,:,0], cmap='gray')
        plt.show()
        
        plt.imshow(images_fake[15,:,:,0], cmap='gray')
        plt.show()
        
        plt.imshow(images_fake[25,:,:,0], cmap='gray')
        plt.show()
    i = i + 1 
    
# Save models 
1/0
disc_net_json = disc_net.to_json()

with open("disc_net.json", "w") as json_file:
    json_file.write(disc_net_json)
    
disc_net.save_weights("disc_net.h5")
print("Saved model to disk")    



    
model_adversarial_json = model_adversarial.to_json()

with open("model_adversarial.json", "w") as json_file:
    json_file.write(model_adversarial_json)
    
model_adversarial.save_weights("model_adversarial_json.h5")
print("Saved model to disk") 