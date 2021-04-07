import tensorflow as tf
import os
import time
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, ReLU, concatenate, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from IPython import display
import cv2
import random

PATH = 'D:/01_data/'

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def load_xy():
    input_image_path = PATH + 'padding_img/'
    target_image_path = PATH + 'resize_img/'
    
    for i in range(30):
        if i==0:
            input_image = cv2.imread(input_image_path+'{}.jpg'.format(str(i)))
            target_image = cv2.imread(target_image_path+'{}.jpg'.format(str(i)))
            input_images = input_image.reshape(1,input_image.shape[0],input_image.shape[1],3)
            target_images = target_image.reshape(1,target_image.shape[0],target_image.shape[1],3)
        else:
            try:
                input_image = cv2.imread(input_image_path+'{}.jpg'.format(str(i)))
                target_image = cv2.imread(target_image_path+'{}.jpg'.format(str(i)))
                input_image = input_image.reshape(1,input_image.shape[0],input_image.shape[1],3)
                target_image = target_image.reshape(1,target_image.shape[0],target_image.shape[1],3)
                input_images = np.concatenate([input_images,input_image])
                target_images = np.concatenate([target_images,target_image])
            except:
                continue
    
    return input_images,target_images

# normalizing the images to [-1, 1]

def normalize(input_image, target_image):
    # 전처리
    for i in range(len(input_image)):
        if i==0:
            input_images = (input_image[i]/ 127.5) - 1
            target_images = (target_image[i]/ 127.5) - 1
            input_images,target_images = input_images.reshape(1,input_images.shape[0],input_images.shape[1],3),target_images.reshape(1,target_images.shape[0],target_images.shape[1],3)
        
        else:
            temp_1 = input_image[i].reshape(1,input_image.shape[1],input_image.shape[2],3)/127.5 - 1
            temp_2 = target_image[i].reshape(1,target_image.shape[1],target_image.shape[2],3)/127.5 - 1
            input_images = np.concatenate([input_images,temp_1])
            target_images = np.concatenate([target_images,temp_2])

    return input_images, target_images

input_images, target_images = load_xy()
print(input_images.shape, target_images.shape)
input_images, target_images = normalize(input_images, target_images)
input_images.shape, target_images.shape

inputs = Input(shape=[720,1280,3])
down_stack0 = Conv2D(64, 4, strides=(2,4), padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(inputs)# (bs, 128, 128, 64)
down_stack1 = Conv2D(128, 4, strides=(3,2), padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(down_stack0)# (bs, 128, 128, 64)
down_stack2 = BatchNormalization()(down_stack1)
down_stack3 = LeakyReLU()(down_stack2)
down_stack4 = Conv2D(256, 4, strides=(3,4), padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(down_stack3)
down_stack5 = BatchNormalization()(down_stack4)
down_stack6 = LeakyReLU()(down_stack5)
down_stack7 = Conv2D(512, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(down_stack6)
down_stack8 = BatchNormalization()(down_stack7)
down_stack9 = LeakyReLU()(down_stack8)
down_stack10 = Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(down_stack9)
down_stack11 = BatchNormalization()(down_stack10)
down_stack12 = LeakyReLU()(down_stack11)
down_stack13 = Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(down_stack12)
down_stack14 = BatchNormalization()(down_stack13)
down_stack15 = LeakyReLU()(down_stack14)
down_stack16 = Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(down_stack15)
down_stack17 = BatchNormalization()(down_stack16)
down_stack18 = LeakyReLU()(down_stack17)
down_stack19 = Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(down_stack18)
down_stack20 = BatchNormalization()(down_stack19)
down_stack21 = LeakyReLU()(down_stack20)
down_stack22 = Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(down_stack21)
down_stack23 = BatchNormalization()(down_stack22)
down_stack24 = LeakyReLU()(down_stack23)

up_stack1 = Conv2DTranspose(1024, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False)(down_stack24)
up_stack2 = Dropout(0.5)(up_stack1)
up_stack3 = ReLU()(up_stack2)
merge1 = concatenate([down_stack21, up_stack3])
up_stack4 = Conv2DTranspose(1024, 2, strides=1,kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False)(merge1)
up_stack5 = Dropout(0.5)(up_stack4)
up_stack6 = ReLU()(up_stack5)
merge2 = concatenate([down_stack18, up_stack6])
up_stack7 = Conv2DTranspose(1024, 3, strides=1,kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False)(merge2)
up_stack8 = Dropout(0.5)(up_stack7)
up_stack9 = ReLU()(up_stack8)
merge3 = concatenate([down_stack15, up_stack9])
up_stack10 = Conv2DTranspose(1024, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False)(merge3)
merge4 = concatenate([down_stack12, up_stack10])
up_stack11 = Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False)(merge4)
merge5 = concatenate([down_stack9, up_stack11])
up_stack12 = Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False)(merge5)
merge6 = concatenate([down_stack6, up_stack12]) 
up_stack13 = Conv2DTranspose(128, 4, strides=(3,4),padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False)(merge6)
merge7 = concatenate([down_stack3, up_stack13])
up_stack14 = Conv2DTranspose(64, 4, strides=(3,2),padding='same',kernel_initializer= tf.random_normal_initializer(0., 0.02),use_bias=False)(merge7)
merge8 = concatenate([down_stack0, up_stack14])
g_out_put = Conv2DTranspose(3, 4,strides=(2,4),padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),activation='tanh')(merge8) # (bs, 256, 256, 3)

generator = Model(inputs = inputs, outputs = g_out_put)
generator.summary()

inp = Input(shape=[720, 1280, 3], name='input_image')
tar = Input(shape=[720, 1280, 3], name='target_image')

merge1 = concatenate([inp, tar])# (bs, 256, 256, channels*2)
discriminator1 = Conv2D(64, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(merge1)
discriminator2 = Conv2D(128, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(discriminator1)
discriminator3 = BatchNormalization()(discriminator2)
discriminator4 = LeakyReLU()(discriminator3)
discriminator5 = Conv2D(256, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(discriminator4)
discriminator6 = BatchNormalization()(discriminator5)
discriminator7 = LeakyReLU()(discriminator6)
discriminator8 = ZeroPadding2D()(discriminator7)
discriminator9 = Conv2D(512, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(discriminator8)
discriminator10 = BatchNormalization()(discriminator9)
discriminator11 = LeakyReLU()(discriminator10)
discriminator12 = ZeroPadding2D()(discriminator11)
discriminator9 = Conv2D(512, 4, strides=2, padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(discriminator8)
discriminator10 = BatchNormalization()(discriminator9)
discriminator11 = LeakyReLU()(discriminator10)
discriminator12 = ZeroPadding2D()(discriminator11)
d_out_put = Conv2D(1, 4, strides=1,kernel_initializer=tf.random_normal_initializer(0., 0.02),activation='sigmoid')(discriminator12)

discriminator = Model(inputs = [inp, tar], outputs = d_out_put)

discriminator.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate=0.0002,beta_1=0.5), metrics=['accuracy'])
discriminator.trainable = False
discriminator.summary()

A = Input(shape=[720, 1280, 3], name='input_image')
B = Input(shape=[720, 1280, 3], name='target_image')
dis_output = [discriminator([generator(B),B], generator(B))]
gen_in = [A, B]
gan = Model(gen_in, dis_output)
gan.compile(loss=['binary_crossentropy','mae'], optimizer=Adam(learning_rate=0.0002,beta_1=0.5))
gan.summary()

accuracies = []
losses = []


def train(epochs, batch_size, sample_interval):


    for epochs in range(epochs):

        # Get a random batch of real images and their labels
        idx = np.random.randint(0, (len(input_images))-1)
        # print('idx = ',idx)
        imgsA = input_images[idx : (idx + 1)]
        imgsB = input_images[idx : (idx + 1)]
        # print('imgb = ',imgsB.shape)
        # imgs1 = np.delete(imgsA, range(idx,(idx+1)),0)
        # imgs2 = np.delete(imgsB, range(idx,(idx+1)),0)
        
        batchA = imgsA.reshape(1,720, 1280, 3)
        batchB = imgsB.reshape(1,720, 1280, 3)
        # real = np.ones((1,91, 161, 1))
        # fake = np.zeros((1,91, 161, 1))
        for b in range(batch_size) :
            real = np.ones((1,)+(int(720/2**4),int(1280/2**4),1))
            fake = np.zeros((1,)+(int(720/2**4),int(1280/2**4),1))

            # Generate a batch of fake images
            gen_imgs = generator.predict(batchB)

            # Train the Discriminator
            d_loss_real = discriminator.train_on_batch([batchA, batchB], real) 
            d_loss_fake = discriminator.train_on_batch([gen_imgs, batchB], fake) 
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the Generator
            g_loss = gan.train_on_batch([batchA, batchB], [real, batchA])
            print('epoch:%d' % epochs, ' d_loss:%.4f' % d_loss, ' g_loss:%.4f' % g_loss)

            if (epochs + 1) % sample_interval == 0:
                # image_grid_rows, image_grid_columns = 3,3
                # random_inds = random.sample(range(len(input_images)),3)
                # imags1 = target_images[random_inds].reshape(3,720,1280,3)
                # imags2 = target_images[random_inds].reshape(3,720,1280,3)
                # gen_img = generator.predict(imags2)
                # gen_img = 0.5*gen_img+0.5
                # fig, axs = plt.subplots(image_grid_rows, image_grid_columns)
                # cnt = 0
                # for i in range(image_grid_rows):
                #     for j in range(image_grid_columns):
                #         # Output a grid of images
                #         axs[i, j].imshow(gen_img[cnt])
                #         axs[i, j].axis('off')
                #         cnt += 1
                #         fig.savefig("D:/gain/"+str(b)+'.png')
                #         plt.close('all')
            
                label = str(epochs)+'_'+str(b)
                plot_chekpoint(label)
                print('batch:'+str(int(b))+',[full doscriminator :: loss : '+str(d_loss)+'],[generator :: loss: '+str(g_loss)+']')
                print('epochs: '+str(int(epochs))+',[full doscriminator :: loss : '+str(d_loss)+'],[generator :: loss: '+str(g_loss)+']')

def plot_chekpoint(b):
    orig_filename = "D:/gain/"+str(b)+'origin.png'
    image_grid_rows, image_grid_columns = 3,3
    random_inds = random.sample(range(len(input_images)),3)
    
    imags1 = input_images[random_inds].reshape(3,720,1280,3)
    imags2 = target_images[random_inds].reshape(3,720,1280,3)
    gen_img = generator.predict(imags1)
    # gen_img = np.concatenate([imags2,fake1,imags1]).reshape(-1,720,1280,3)
    gen_img = 0.5*gen_img+0.5


    titles = ['style','generated','origin']
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_img[cnt])
            axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
            fig.savefig("D:/gain/"+str(b)+'.png')
            plt.close('all')
            return



# Set hyperparameters
epochs = 1001
batch_size = 1
sample_interval = 10

train(epochs, batch_size, sample_interval)




def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (100 * l1_loss)

    return total_gen_loss, gan_loss, l1_loss



# def generate_images(model, test_input, tar):
#   prediction = model(test_input, training=True)
#   plt.figure(figsize=(15,15))

#   display_list = [test_input[0], tar[0], prediction[0]]
#   title = ['Input Image', 'Ground Truth', 'Predicted Image']

#   fig, axs = plt.subplots(5, 5)

#   for i in range(3):
#     plt.subplot(1, 3, i+1)
#     plt.title(title[i])
#     # # getting the pixel values between [0, 1] to plot it.
#     gen_imgs = display_list[i] * 0.5 + 0.5
#     plt.imshow(gen_imgs)
#     # plt.axis('off')
#     for j in range(3):
#             # 이미지 그리드 출력
#             axs[i, j].imshow(gen_imgs[cnt, :, :, ])
#             axs[i, j].axis('off')
#   count = count + 1
#   fig.savefig("D:/gain/gan_" + str(count) + ".png" )
#   # plt.show()


# for example_input, example_target in test_dataset.take(1):
#   generate_images(generator, example_input, example_target)

# EPOCHS = 5000

# import datetime
# log_dir="logs/"

# summary_writer = tf.summary.create_file_writer(
#   log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# @tf.function
# def train_step(input_image, target, epoch):
#   with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#     gen_output = generator(input_image, training=True)

#     disc_real_output = discriminator([input_image, target], training=True)
#     disc_generated_output = discriminator([input_image, gen_output], training=True)

#     gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
#     disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

#   generator_gradients = gen_tape.gradient(gen_total_loss,
#                                           generator.trainable_variables)
#   discriminator_gradients = disc_tape.gradient(disc_loss,
#                                                discriminator.trainable_variables)

#   generator_optimizer.apply_gradients(zip(generator_gradients,
#                                           generator.trainable_variables))
#   discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
#                                               discriminator.trainable_variables))

#   with summary_writer.as_default():
#     tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
#     tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
#     tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
#     tf.summary.scalar('disc_loss', disc_loss, step=epoch)


# def fit(train_ds, epochs, test_ds):
#     for epoch in range(epochs):
#         start = time.time()

#         display.clear_output(wait=True)

#         for example_input, example_target in test_ds.take(1):
#             generate_images(generator, example_input, example_target)
#         print("Epoch: ", epoch)

#         # Train
#         for n, (input_image, target) in train_ds.enumerate():
#             print('.', end='')
#             if (n+1) % 100 == 0:
#                 print()
#             train_step(input_image, target, epoch)
#         print()

#         # saving (checkpoint) the model every 20 epochs
#         if (epoch + 1) % 20 == 0:
#             checkpoint.save(file_prefix = checkpoint_prefix)

#         print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
#                                                             time.time()-start))
#     checkpoint.save(file_prefix = checkpoint_prefix)

# fit(train_dataset, EPOCHS, test_dataset)