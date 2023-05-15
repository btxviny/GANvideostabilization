import numpy as np
import keras
import tensorflow as tf
from keras.layers import Conv2D,AveragePooling2D,Concatenate,UpSampling2D,Input,LeakyReLU,Flatten,Dense,BatchNormalization,Dropout
import cv2

class Localization(tf.keras.layers.Layer):
    def __init__(self):
        super(Localization, self).__init__()
        self.pool1 = tf.keras.layers.MaxPool2D(trainable=True)
        self.conv1 = tf.keras.layers.Conv2D(16, [5, 5], activation='relu', trainable=True)
        self.pool2 = tf.keras.layers.MaxPool2D(trainable=True)
        self.conv2 = tf.keras.layers.Conv2D(20, [5, 5], activation='relu', trainable=True)
        self.flatten = tf.keras.layers.Flatten(trainable=True)
        self.fc1 = tf.keras.layers.Dense(6, activation=None, bias_initializer=tf.keras.initializers.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), kernel_initializer='zeros', trainable=True)


    def build(self, input_shape):
        print('Building STN Localization Network')
        super(Localization,self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return [None, 6]

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        theta = self.fc1(x)
        theta = tf.keras.layers.Reshape((2, 3))(theta)
        return theta
    
class BilinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, height=40, width=40):
        super(BilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, 1]

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
        }
    
    def build(self, input_shape):
        return

    def advance_indexing(self, inputs, x, y):
        '''
        Numpy like advance indexing is not supported in tensorflow, hence, this function is a hack around the same method
        '''        
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2]
        
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(inputs, indices)

    def call(self, inputs):
        images, theta = inputs
        homogenous_coordinates = self.grid_generator(batch=tf.shape(images)[0])
        return self.interpolate(images, homogenous_coordinates, theta)

    def grid_generator(self, batch):
        x = tf.linspace(-1, 1, self.width)
        y = tf.linspace(-1, 1, self.height)
            
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates
    
    def interpolate(self, images, homogenous_coordinates, theta):

        with tf.name_scope("Transformation"):
            transformed = tf.matmul(theta, homogenous_coordinates)
            transformed = tf.transpose(transformed, perm=[0, 2, 1])
            transformed = tf.reshape(transformed, [-1, self.height, self.width, 2])
                
            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]
                
            x = ((x_transformed + 1.) * tf.cast(self.width, dtype=tf.float32)) * 0.5
            y = ((y_transformed + 1.) * tf.cast(self.height, dtype=tf.float32)) * 0.5

        with tf.name_scope("VariableCasting"):
            x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, self.width-1)
            x1 = tf.clip_by_value(x1, 0, self.width-1)
            y0 = tf.clip_by_value(y0, 0, self.height-1)
            y1 = tf.clip_by_value(y1, 0, self.height-1)
            x = tf.clip_by_value(x, 0, tf.cast(self.width, dtype=tf.float32)-1.0)
            y = tf.clip_by_value(y, 0, tf.cast(self.height, dtype=tf.float32)-1)

        with tf.name_scope("AdvanceIndexing"):
            Ia = self.advance_indexing(images, x0, y0)
            Ib = self.advance_indexing(images, x0, y1)
            Ic = self.advance_indexing(images, x1, y0)
            Id = self.advance_indexing(images, x1, y1)

        with tf.name_scope("Interpolation"):
            x0 = tf.cast(x0, dtype=tf.float32)
            x1 = tf.cast(x1, dtype=tf.float32)
            y0 = tf.cast(y0, dtype=tf.float32)
            y1 = tf.cast(y1, dtype=tf.float32)
                            
            wa = (x1-x) * (y1-y)
            wb = (x1-x) * (y-y0)
            wc = (x-x0) * (y1-y)
            wd = (x-x0) * (y-y0)

            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)
                        
        return tf.math.add_n([wa*Ia + wb*Ib + wc*Ic + wd*Id])    



def build_generator(h,w):
    initializer = tf.keras.initializers.RandomNormal(mean = 1, stddev=0.1,seed=42)

    input_sequence = Input(shape =(h,w,5))
    input_frame = Input(shape = (h,w,3))
    #####################
    # ENCODER 
    ######################
    concatenated_0 = Concatenate(axis = -1)([input_sequence, input_frame])
    theta_0 = Localization()(concatenated_0)
    warped_0 = BilinearInterpolation(256,256)([input_frame,theta_0])

    conv_10 =  Conv2D(64, kernel_size=(3,3),strides=(2,2), name='conv_10', padding='same', kernel_initializer=initializer)(warped_0)
    conv_10 = LeakyReLU(0.2)(conv_10)

    conv_11 = Conv2D(64, kernel_size=(3,3),strides=(2,2), name='conv_11',padding='same', kernel_initializer=initializer)(input_sequence)
    conv_11 = LeakyReLU(0.2)(conv_11)
    concatenated_1 = Concatenate(axis = -1)([conv_10, conv_11])
    theta_1 = Localization()(concatenated_1)
    warped_1 = BilinearInterpolation(128,128)([conv_10, theta_1])

    conv_20 =  Conv2D(128, kernel_size=(3,3),strides=(2,2), name='conv_20',padding='same', kernel_initializer=initializer)(warped_1) #####
    conv_20 = LeakyReLU(0.2)(conv_20)
    conv_21 = Conv2D(128, kernel_size=(3,3),strides=(2,2), name='conv_21',padding='same', kernel_initializer=initializer)(conv_11)
    conv_21 = LeakyReLU(0.2)(conv_21)
    concatenated_2 = Concatenate(axis = -1)([conv_20, conv_21])
    theta_2 = Localization()(concatenated_2)
    warped_2 = BilinearInterpolation(64,64)([conv_20,theta_2])

    conv_30 = Conv2D(256, kernel_size=(3,3),strides=(2,2), name='conv_30',padding='same', kernel_initializer=initializer)(warped_2) #####
    conv_30 = LeakyReLU(0.2)(conv_30)
    conv_31 = Conv2D(256,(3,3),strides=(2,2), name='conv_31',padding='same', kernel_initializer=initializer)(conv_21)
    conv_31 = LeakyReLU(0.2)(conv_31)
    concatenated_3 = Concatenate(axis=-1)([conv_30, conv_31])
    theta_3 = Localization()(concatenated_3)
    warped_3 = BilinearInterpolation(32,32)([conv_30,theta_3])

    conv_40 = Conv2D(512, (3,3),strides=(2,2),name='conv_40',padding='same', kernel_initializer=initializer)(warped_3)
    conv_40 = LeakyReLU(0.2)(conv_40)
    conv_41 = Conv2D(512, (3,3),strides=(2,2),name='conv_41',padding='same', kernel_initializer=initializer)(conv_31)
    conv_41 = LeakyReLU(0.2)(conv_41)
    concatenated_4 = tf.keras.layers.Concatenate(axis =-1)([conv_40, conv_41])
    theta_4 = Localization()(concatenated_4)
    warped_4 = BilinearInterpolation(16,16)([conv_40, theta_4])

    conv_5 = Conv2D(512, (3,3),strides=(2,2),name='conv_5',padding='same', kernel_initializer=initializer)(warped_4)
    conv_5 = LeakyReLU(0.2)(conv_5)
    conv_6 = Conv2D(512, (3,3),strides=(2,2),name='conv_6',padding='same', kernel_initializer=initializer)(conv_5)
    conv_6 = LeakyReLU(0.2)(conv_6)
    conv_7 = Conv2D(512, (3,3),strides=(2,2),name='conv_7',padding='same', kernel_initializer=initializer)(conv_6)
    conv_7 = LeakyReLU(0.2)(conv_7)
    ##############################################
    #DECODER
    ##############################################
    up_1 = UpSampling2D(size=(2,2))(conv_7)
    concatenated_5 = Concatenate(axis = -1)([up_1, conv_6])
    deconv_1 = Conv2D(1024, (3,3), activation= 'relu',padding ='same',  name = 'deconv_1', kernel_initializer=initializer)(concatenated_5)

    up_2 = UpSampling2D(size=(2,2))(deconv_1)
    concatenated_6 = Concatenate(axis=-1)([up_2, conv_5])
    deconv_2 = Conv2D(512, (3,3),activation= 'relu',padding ='same', name = 'deconv_2' , kernel_initializer=initializer)(concatenated_6)

    up_3 = UpSampling2D(size=(2,2))(deconv_2)
    concatenated_7 = Concatenate(axis=-1)([up_3, conv_40])
    deconv_3 = Conv2D(512, (3,3),activation= 'relu',padding ='same', name = 'deconv_3', kernel_initializer=initializer)(concatenated_7)

    up_4 = UpSampling2D(size=(2,2))(deconv_3)
    concatenated_8 = Concatenate(axis=-1)([up_4, conv_30])
    deconv_4 = Conv2D(256, (3,3),activation= 'relu',padding ='same', name = 'deconv_4', kernel_initializer=initializer)(concatenated_8)

    up_5 = UpSampling2D(size=(2,2))(deconv_4)
    concatenated_9 = Concatenate(axis=-1)([up_5, conv_20])
    deconv_5 = Conv2D(128, (3,3),activation= 'relu',padding ='same', name = 'deconv_5', kernel_initializer=initializer)(concatenated_9)

    up_6 = UpSampling2D(size=(2,2))(deconv_5)
    concatenated_10 = Concatenate(axis=-1)([up_6, conv_10])
    deconv_6 =Conv2D(64, (3,3),activation= 'relu',padding ='same', name = 'deconv_6', kernel_initializer=initializer)(concatenated_10)

    up_7 = UpSampling2D(size=(2,2))(deconv_6)
    output_uncroped = Conv2D(3,(3,3),activation ='tanh',padding ='same', name = 'out', kernel_initializer=initializer)(up_7)

    Autoencoder = tf.keras.Model(inputs = [input_sequence , input_frame], outputs = [output_uncroped,[warped_0, warped_1, warped_2, warped_3,warped_4]])
    return(Autoencoder)


def build_discriminator(input_shape):
    initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev=0.02,seed=42)
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dropout(0.25)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dropout(0.25)(x)
    x = Conv2D(256, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dropout(0.25)(x)
    x = Conv2D(512, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(1, kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

class SaveModelsCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator, d1, d2, save_freq, path):
        super(SaveModelsCallback, self).__init__()
        self.generator = generator
        self.d1 = d1
        self.d2 = d2
        self.save_freq = save_freq
        self.path = path
        self.epoch_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        if self.epoch_count % self.save_freq == 0:
            generator_file_name = self.path + "/generator.h5"
            d1_file_name = self.path + "/d1.h5"
            d2_file_name = self.path + "/d2.h5"
            self.generator.save(generator_file_name)
            self.d1.save(d1_file_name)
            self.d2.save(d2_file_name)

