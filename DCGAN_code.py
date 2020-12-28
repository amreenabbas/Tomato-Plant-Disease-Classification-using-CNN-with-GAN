# define the standalone discriminator model 
def define_discriminator(in_shape=(128,128,3)):
  model = Sequential()
  #Normal
  model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape)) 
  model.add(LeakyReLU(alpha=0.2))
  #Downsample
  model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  #Downsample
  model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  #Downsample
  model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  #Downsample
  model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  #Downsample
  model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  #classifier
  model.add(Flatten()) 
  model.add(Dropout(0.4)) 
  model.add(Dense(1, activation='sigmoid')) 
  #compile model
  opt = Adam(lr=0.0002, beta_1=0.5) 
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) 
  return model

# define model 
model = define_discriminator() 
# summarize the model 
model.summary() 




# define the standalone generator model 
def define_generator(latent_dim): 
  model = Sequential() 
  # foundation for 4x4 image 
  n_nodes = 256 * 4 * 4 
  model.add(Dense(n_nodes, input_dim=latent_dim)) 
  model.add(LeakyReLU(alpha=0.2)) 
  model.add(Reshape((4, 4, 256))) 
  # upsample to 8x8 
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) 
  model.add(LeakyReLU(alpha=0.2)) 
  # upsample to 16x16 
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) 
  model.add(LeakyReLU(alpha=0.2))
  # upsample to 32x32 
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2)) 
  # upsample to 64x64 
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) 
  model.add(LeakyReLU(alpha=0.2)) 
  # upsample to 128x128 
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) 
  model.add(LeakyReLU(alpha=0.2)) 
  # output layer 
  model.add(Conv2D(3, (3,3), activation='tanh', padding='same')) 
  return model
  model.add(Conv2D(128, padding="same", kernel_size=3))

# define the size of the latent space 
latent_dim = 100 
# define the generator model 
model = define_generator(latent_dim) 
# summarize the model 
model.summary() 




def define_gan(g_model, d_model): 
  # make weights in the discriminator not trainable 
  d_model.trainable = False 
  # connect them 
  model = Sequential() 
  # add generator 
  model.add(g_model) 
  # add the discriminator 
  model.add(d_model) 
  # compile model 
  opt = Adam(lr=0.0002, beta_1=0.5) 
  model.compile(loss='binary_crossentropy', optimizer=opt) 
  return model

# size of the latent space 
latent_dim = 100 
# create the discriminator 
d_model = define_discriminator() 
# create the generator 
g_model = define_generator(latent_dim) 
# create the gan 
gan_model = define_gan(g_model, d_model) 
# summarize gan model 
gan_model.summary() 




# load and prepare training images 
def load_real_samples():
  # convert from unsigned ints to floats 
  X = X_train.astype('float32') 
  # scale from [0,255] to [-1,1] 
  X = (X - 127.5) / 127.5 
  return X




# select real samples 
def generate_real_samples(dataset, n_samples): 
  # choose random instances
  ix = randint(0, dataset.shape[0], n_samples) 
  # retrieve selected images 
  X = dataset[ix] 
  # generate 'real' class labels (1)
  y = ones((n_samples, 1)) 
  return X, y




# generate points in latent space as input for the generator 
def generate_latent_points(latent_dim, n_samples):
   # generate points in the latent space 
   x_input = randn(latent_dim * n_samples) 
   # reshape into a batch of inputs for the network 
   x_input = x_input.reshape(n_samples, latent_dim) 
   return x_input




# generate n fake samples with class labels 
def generate_fake_samples(g_model, latent_dim, n_samples): 
  # generate points in latent space
  x_input = generate_latent_points(latent_dim, n_samples) 
  # predict outputs 
  X = g_model.predict(x_input) 
  # create 'fake' class labels (0) 
  y = zeros((n_samples, 1)) 
  return X, y




# create and save a plot of generated images 
def save_plot(examples, epoch, n=10): 
  # scale from [-1,1] to [0,1] 
  examples = (examples + 1) / 2.0 
  # plot images 
  for i in range(n * n): 
    # define subplot 
    plt.subplot(n, n, 1 + i) 
    # turn off axis 
    plt.axis('off') 
    # plot raw pixel data 
    plt.imshow(examples[i]) 
  # save plot to file 
  filename = 'generated_plot_e%03d.png' % (epoch+1)
  plt.savefig(filename) 
  plt.close()




# evaluate the discriminator, plot generated images, save generator model 
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150): 
  # prepare real samples 
  X_real, y_real = generate_real_samples(dataset, n_samples) 
  # evaluate discriminator on real examples 
  _, acc_real = d_model.evaluate(X_real, y_real, verbose=0) 
  # prepare fake examples 
  x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples) 
  # evaluate discriminator on fake examples 
  _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0) 
  # summarize discriminator performance 
  print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
  # save plot 
  save_plot(x_fake, epoch) 
  # save the generator model tile file 
  filename = 'generator_model_%03d.h5' % (epoch+1) 
  g_model.save(filename)




# train the generator and discriminator 
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128): 
  bat_per_epo = int(dataset.shape[0] / n_batch) 
  half_batch = int(n_batch / 2) 
  # manually enumerate epochs 
  for i in range(n_epochs): 
    # enumerate batches over the training set 
    for j in range(bat_per_epo):
      # get randomly selected 'real' samples 
      X_real, y_real = generate_real_samples(dataset, half_batch) 
      # update discriminator model weights 
      d_loss1, _ = d_model.train_on_batch(X_real, y_real) 
      # generate 'fake' examples 
      X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
      # update discriminator model weights 
      d_loss2, _ = d_model.train_on_batch(X_fake, y_fake) 
      # prepare points in latent space as input for the generator 
      X_gan = generate_latent_points(latent_dim, n_batch) 
      # create inverted labels for the fake samples 
      y_gan = ones((n_batch, 1)) 
      # update the generator via the discriminator's error 
      g_loss = gan_model.train_on_batch(X_gan, y_gan) 
      # summarize loss on this batch 
      print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
    # evaluate the model performance, sometimes 
    if (i+1) % 10 == 0: 
      summarize_performance(i, g_model, d_model, dataset, latent_dim)




# size of the latent space 
latent_dim = 100 
# create the discriminator 
d_model = define_discriminator() 
# create the generator 
g_model = define_generator(latent_dim) 
# create the gan 
gan_model = define_gan(g_model, d_model) 
# summarize gan model 
gan_model.summary() 
# load image data 
dataset = load_real_samples() 
# train model 
train(g_model, d_model, gan_model, dataset, latent_dim)




save_plot(X_train, 1, 10)
plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
plt.show()



