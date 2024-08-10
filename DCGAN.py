from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU, Conv2D, Dropout, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

rows = 28
cols = 28
colors = 1

number = 6

shape = (rows,cols,colors)

def build_generator():

    noise_shape = (100,) #1D array of size 100 (latent vector / noise) 

    model = Sequential()
    
    model.add(Dense(7*7*1000, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization(momentum = 0.1))
    model.add(LeakyReLU(alpha = 0.1))

    model.add(Reshape((7, 7, 1000)))
    assert model.output_shape == (None, 7, 7, 1000)  # Note: None is the batch size

    model.add(Conv2DTranspose(500, (10, 10), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 500)
    model.add(BatchNormalization(momentum = 0.1))
    model.add(LeakyReLU(alpha = 0.1))

    model.add(Conv2DTranspose(150, (10, 10), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 150)
    model.add(BatchNormalization(momentum = 0.1))
    model.add(LeakyReLU(alpha = 0.1))

    model.add(Conv2DTranspose(1, (10, 10), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)    #Generated image

    return Model(noise, img)

def build_discriminator():
    model = Sequential()

    model.add(Flatten(input_shape=shape))
    model.add(Dense(1000))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(500))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=shape)
    validity = model(img)

    return Model(img, validity)


def train(epochs, batch_size=128, save_interval=1000):

    # Load the dataset
    (X_train, Y_label), (_, _) = mnist.load_data()

    # Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    #Select a particular number; for this example we use 5
    indices = np.where(Y_label == number) [0]
    X_train = X_train[indices]    

#Add channels dimension. As the input to our gen and discr. has a shape 28x28x1.
    
    
    X_train = np.expand_dims(X_train, axis=3) 

    half_batch = int(batch_size / 2)


#We then loop through a number of epochs to train our Discriminator by first selecting
#a random batch of images from our true dataset, generating a set of images from our
#Generator, feeding both set of images into our Discriminator, and finally setting the
#loss parameters for both the real and fake images, as well as the combined loss. 
    
    for epoch in range(epochs):


        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]

 
        noise = np.random.normal(0, 1, (half_batch, 100))

        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 

        noise = np.random.normal(0, 1, (batch_size, 100)) 

        valid_y = np.array([1] * batch_size) #Creates an array of all ones of size=batch size

        g_loss = combined.train_on_batch(noise, valid_y)

        
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch)

def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    plt.savefig("mnist_%d.png" % epoch)
    plt.close()

optimizer = Adam(0.0001, 0.5)  #Learning rate and momentum.

# Build and compile the discriminator first. 
#Generator will be trained as part of the combined model, later. 
#pick the loss function and the type of metric to keep track.                 
#Binary cross entropy as we are doing prediction and it is a better
#loss function compared to MSE or other. 
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

#build and compile our Discriminator, pick the loss function

#SInce we are only generating (faking) images, let us not track any metrics.
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

##This builds the Generator and defines the input noise. 
#In a GAN the Generator network takes noise z as an input to produce its images.  
z = Input(shape=(100,))   #Our random input to the generator
img = generator(z)

#This ensures that when we combine our networks we only train the Generator.
#While generator training we do not want discriminator weights to be adjusted. 
#This Doesn't affect the above descriminator training.     
discriminator.trainable = False  

#This specifies that our Discriminator will take the images generated by our Generator
#and true dataset and set its output to a parameter called valid, which will indicate
#whether the input is real or not.  
valid = discriminator(img)  #Validity check on the generated image


#Here we combined the models and also set our loss function and optimizer. 
#Again, we are only training the generator here. 
#The ultimate goal here is for the Generator to fool the Discriminator.  
# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


train(epochs=10000, batch_size=32, save_interval=500)

#Save model for future use to generate fake images
#Not tested yet... make sure right model is being saved..
#Compare with GAN4

generator.save('generator_model_DCGAN_'+str(number)+'.h5')  #Test the model on GAN4_predict...
#Change epochs back to 30K
                
#Epochs dictate the number of backward and forward propagations, the batch_size
#indicates the number of training samples per backward/forward propagation, and the
#sample_interval specifies after how many epochs we call our sample_image function.