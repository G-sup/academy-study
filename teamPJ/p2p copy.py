import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
import os
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, ReLU, concatenate, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from keras.preprocessing import image



_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


inp, re = load(PATH+'train/100.jpg')
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp/255.0)
plt.figure()
plt.imshow(re/255.0)

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
  stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image


plt.figure(figsize=(6, 6))
for i in range(4):
  rj_inp, rj_re = random_jitter(inp, re)
  plt.subplot(2, 2, i+1)
  plt.imshow(rj_inp/255.0)
  plt.axis('off')
plt.show()

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)

down_stack = Sequential()      

down_stack.add(Conv2D(64, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False,input_shape=(256,256,3)))
down_stack.add(Conv2D(128, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
down_stack.add(BatchNormalization())
down_stack.add(LeakyReLU())
down_stack.add(Conv2D(256, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
down_stack.add(BatchNormalization())
down_stack.add(LeakyReLU())
down_stack.add(Conv2D(512, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
down_stack.add(BatchNormalization())
down_stack.add(LeakyReLU())
down_stack.add(Conv2D(512, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
down_stack.add(BatchNormalization())
down_stack.add(LeakyReLU())
down_stack.add(Conv2D(512, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
down_stack.add(BatchNormalization())
down_stack.add(LeakyReLU())
down_stack.add(Conv2D(512, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
down_stack.add(BatchNormalization())
down_stack.add(LeakyReLU())
down_stack.add(Conv2D(512, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
down_stack.add(BatchNormalization())
down_stack.add(LeakyReLU())

up_stack = Sequential()      
up_stack.add(Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False))
up_stack.add(Dropout(0.5))
up_stack.add(ReLU())
up_stack.add(Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False))
up_stack.add(Dropout(0.5))
up_stack.add(ReLU())
up_stack.add(Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False))
up_stack.add(Dropout(0.5))
up_stack.add(ReLU())
up_stack.add(Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False))
up_stack.add(Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False))
up_stack.add(Conv2DTranspose(128, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False))
up_stack.add(Conv2DTranspose(64, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False))
up_stack.add(Conv2DTranspose(3, 4,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),activation='tanh')) # (bs, 256, 256, 3)
up_stack.summary()

down_stack = reversed([down_stack])

generator.Concatenate([up_stack, down_stack])
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

gen_output = generator(inp[tf.newaxis,...], training=False)
plt.imshow(gen_output[0,...])

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss





discriminator = Sequential()
inp = discriminator.add(Input(shape=[256, 256, 3], name='input_image'))
tar = discriminator.add(Input(shape=[256, 256, 3], name='target_image'))

discriminator.add(concatenate([inp, tar])) # (bs, 256, 256, channels*2)
discriminator.add(Conv2D(64, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False,input_shape=(256,256,3)))
discriminator.add(Conv2D(128, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU())
discriminator.add(Conv2D(256, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU())
discriminator.add(ZeroPadding2D())
discriminator.add(Conv2D(512, 4, strides=1, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU())
discriminator.add(ZeroPadding2D())
discriminator.add(Conv2D(1, 4, strides=1,kernel_initializer=tf.random_normal_initializer(0., 0.02)))

ginput = Input(shape=[256,256,3])
dis_output = discriminator(generator(ginput))
gan = Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002,beta_1=0.5))
gan.summary()


count = 0

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  fig, axs = plt.subplots(5, 5)
  cnt = 0
  global count

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # # getting the pixel values between [0, 1] to plot it.
    gen_imgs = display_list[i] * 0.5 + 0.5
    plt.imshow(gen_imgs)
    # plt.axis('off')
    for j in range(3):
            # 이미지 그리드 출력
            axs[i, j].imshow(gen_imgs[cnt, :, :, ])
            axs[i, j].axis('off')
  count = count + 1
  fig.savefig("D:/gain/gan_" + str(count) + ".png" )
  # plt.show()


for example_input, example_target in test_dataset.take(1):
  generate_images(generator, example_input, example_target)

EPOCHS = 5000

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))
    checkpoint.save(file_prefix = checkpoint_prefix)

fit(train_dataset, EPOCHS, test_dataset)