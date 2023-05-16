import numpy as np
import keras
import tensorflow as tf
from keras.layers import Conv2D,MaxPool2D,Concatenate,UpSampling2D,Input,LeakyReLU,Flatten,Dense,BatchNormalization,Dropout
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

class STN(tf.keras.Model):
    def __init__(self,shape):
        super(STN, self).__init__()
        self.Localization = Localization()
        self.BilinearInterpolation =  BilinearInterpolation(shape[0],shape[1])
    def call(self, inputs):
        feat1, feat2 = inputs
        theta = self.Localization(tf.concat([feat1, feat2],axis =-1))
        warped = self.BilinearInterpolation([feat1,theta])
        #warped = tfa.image.transform(feat1,theta,interpolation='nearest')
        return warped, theta

initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev=0.002,seed=42)
class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet,self).__init__()
        class Encoder(tf.keras.layers.Layer):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=(1,1)):
                super(Encoder, self).__init__()

                self.seq = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(out_nc, kernel_size=k_size, strides=stride, padding='same',kernel_initializer=initializer),
                    tf.keras.layers.LeakyReLU(0.2)
                ])

            def call(self, x):
                feature = self.seq(x)
                return feature


        class Decoder(tf.keras.layers.Layer):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
                super(Decoder, self).__init__()

                self.seq = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(in_nc, kernel_size=k_size, strides=stride, padding='same',kernel_initializer=initializer),
                    tf.keras.layers.Conv2D(out_nc, kernel_size=k_size, strides=stride, padding='same',kernel_initializer=initializer)
                ])

                if tanh:
                    self.activ = tf.keras.layers.Activation('tanh')
                else:
                    self.activ = tf.keras.layers.ReLU()

            def call(self, x):
                s = self.seq(x)
                s = self.activ(s)
                return s 
            
        #Encoder
        self.stn0 = STN((256,256))
        self.enc10 = Encoder(3,64,k_size=5, stride=2)
        self.enc11 = Encoder(5, 64,k_size=5, stride=2)
        #-------------------------------------
        self.stn1 = STN((128,128))
        self.enc20 = Encoder(64,128,k_size=5,stride=2)
        self.enc21 = Encoder(64,128,k_size=5,stride=2)
        #-------------------------------------
        self.stn2 = STN((64,64))
        self.enc30 = Encoder(128,256,k_size=3,stride=2)
        self.enc31 = Encoder(128,256,k_size=3,stride=2)
        #-------------------------------------
        self.stn3 = STN((32,32))
        self.enc40 = Encoder(256,512,k_size=1,stride=2)
        self.enc41 = Encoder(256,512,k_size=1,stride=2)
        #-------------------------------------
        self.stn4 = STN((16,16))
        self.enc5 = Encoder(512,512,k_size=1,stride=2)
        self.enc6 = Encoder(512,512,k_size=1,stride=2)
        self.enc7 = Encoder(512,512,k_size=1,stride = 2)
        #Decoder
        self.dec0 = Decoder(512, 1024,k_size=3, stride=1)
        self.dec1 = Decoder(1024, 512,k_size=3, stride=1)
        self.dec2 = Decoder(512, 512,k_size=3, stride=1)
        self.dec3 = Decoder(512, 256,k_size=3, stride=1)
        self.dec4 = Decoder(256, 128,k_size=5, stride=1)
        self.dec5 = Decoder(128, 64,k_size=5, stride=1 )
        self.dec6 = Decoder(64, 3,k_size=3, stride=1 ,tanh=True)
    def transform(self, input, theta, layer):
        for i in range(layer):
            mat = theta[layer - i - 1]
            input = BilinearInterpolation(input.shape[1],input.shape[2])([input,mat])
        return(input)
    
    def call(self,input):
        sequence, unsteady = input
        
    
        T0, theta0 = self.stn0([unsteady,sequence])
        s10 = self.enc10(T0)
        s11 = self.enc11(sequence)
        #-------------------------------------------------
        T1, theta1 = self.stn1([s10,s11])
        s20 = self.enc20(T1)
        s21 = self.enc21(s11)
        #-------------------------------------------------
        T2, theta2 = self.stn2([s20,s21])
        s30 = self.enc30(T2)
        s31 = self.enc31(s21)
        #-------------------------------------------------
        T3, theta3 = self.stn3([s30,s31])
        s40 = self.enc40(T3)
        s41 = self.enc41(s31)
        #-------------------------------------------------
        T4, theta4 = self.stn4([s40,s41])
        s5 = self.enc5(T4)
        s6 = self.enc6(s5)
        s7 = self.enc7(s6)
        T = [theta4, theta3 ,theta2 ,theta1, theta0]

        up0 = tf.keras.layers.UpSampling2D(size =(2,2),interpolation='nearest')(s7)
        dec0 = self.dec0(tf.concat([up0,s6],axis=-1))
        up1 = tf.keras.layers.UpSampling2D(size =(2,2),interpolation='nearest')(dec0)
        dec1 = self.dec1(tf.concat([up1,s5],axis=-1))
        up2 = tf.keras.layers.UpSampling2D(size =(2,2),interpolation='nearest')(dec1)
        trans1 = self.transform(s40,T,1) # warp by T4
        dec2 = self.dec2(tf.concat([up2,trans1],axis=-1))
        up3 = tf.keras.layers.UpSampling2D(size =(2,2),interpolation='nearest')(dec2)
        trans2 = self.transform(s30,T,2)  # warp by T3 X T4
        dec3 = self.dec3(tf.concat([up3,trans2],axis=-1))
        up4 = tf.keras.layers.UpSampling2D(size =(2,2),interpolation='nearest')(dec3)
        trans3 = self.transform(s20,T,3) # warp by T2 X T3 X T4
        dec4 = self.dec4(tf.concat([up4,trans3],axis=-1))
        up5 = tf.keras.layers.UpSampling2D(size =(2,2),interpolation='nearest')(dec4)
        trans4 = self.transform(s10,T,4) # warp by T1 X T2 X T3 X T4
        dec5 = self.dec5(tf.concat([up5,trans4],axis=-1))
        up6 = tf.keras.layers.UpSampling2D(size =(2,2),interpolation='nearest')(dec5)
        out  = self.dec6(up6)
        return out , T

def build_discriminator(input_shape):
    initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev=0.02,seed=42)
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dropout(0.25)(x)
    x = Conv2D(128, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dropout(0.25)(x)
    x = Conv2D(256, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dropout(0.25)(x)
    x = Conv2D(512, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer)(x)
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

