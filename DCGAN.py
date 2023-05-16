import keras
import tensorflow as tf
import numpy as np
import cv2

class GAN(keras.Model):
    def __init__(self, d1 , d2, generator ,A ):
        super().__init__()
        self.d1 = d1
        self.d2= d2
        self.generator = generator
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.d1_loss_tracker = keras.metrics.Mean(name="d1_loss")
        self.d2_loss_tracker = keras.metrics.Mean(name="d2_loss")
        self.vgg = tf.keras.applications.VGG19(include_top = False, weights='imagenet')
        self.vgg.trainable = False
        self.A = tf.Variable(initial_value=A, trainable=False)
    @property
    def metrics(self):
        return [self.gen_loss_tracker,  self.d1_loss_tracker,  self.d2_loss_tracker]

    def compile(self, d1_optimizer, d2_optimizer, g_optimizer):
        super().compile()
        self.d1_optimizer = d1_optimizer
        self.d2_optimizer = d2_optimizer
        self.gen_optimizer = g_optimizer
    
    def train_step(self, data): 
        #Interact with Datagenerator
        with tf.GradientTape() as d1_tape, tf.GradientTape() as d2_tape, tf.GradientTape() as gen_tape:
             # Unpack the data.
            s , It, Igt  = data
            generated_frame, _ = self.generator([s, It])
            
            real_in_d1 = tf.concat([s, Igt], axis=-1)
            fake_in_d1 = tf.concat([s, generated_frame], axis=-1)

            real_in_d2 = tf.concat([self.A, Igt], axis=-1)
            fake_in_d2 = tf.concat([self.A, generated_frame], axis=-1)
            
            d1_real_predictions = self.d1(real_in_d1)
            d1_fake_predictions = self.d1(fake_in_d1)
            d2_real_predictions = self.d2(real_in_d2)
            d2_fake_predictions = self.d2(fake_in_d2)

            random_zero = tf.random.uniform(tf.shape(d1_real_predictions), minval=0.0, maxval=0.1)
            random_one = tf.random.uniform(tf.shape(d1_fake_predictions), minval=0.9, maxval=1.0)

            weight = tf.constant([0.5], dtype = tf.float32)
            real_loss_d1 = tf.multiply(weight,tf.reduce_mean(tf.square(random_zero - d1_real_predictions)))
            fake_loss_d1 = tf.multiply(weight,tf.reduce_mean(tf.square(random_one - d1_fake_predictions)))
            weight = tf.constant([0.5], dtype = tf.float32)
            d1_loss = tf.add(real_loss_d1,fake_loss_d1)

            real_loss_d2 = tf.multiply(weight,tf.reduce_mean(tf.square(random_zero - d2_real_predictions)))
            fake_loss_d2 = tf.multiply(weight,tf.reduce_mean(tf.square(random_one - d2_fake_predictions)))
            d2_loss = tf.add(real_loss_d2,fake_loss_d2)

            gen_loss = self.generator_loss([generated_frame, Igt], [d1_fake_predictions, d2_fake_predictions])

            # Watch trainable variables
            d1_tape.watch(self.d1.trainable_variables)
            d2_tape.watch(self.d2.trainable_variables)
            gen_tape.watch(self.generator.trainable_variables)

            #update A sequence
            slice = tf.slice(self.A, [0,0,0,0],[-1,-1,-1,12])
            self.A.assign(tf.concat([generated_frame,slice], axis = -1))

        d1_grads = d1_tape.gradient(d1_loss, self.d1.trainable_weights)
        self.d1_optimizer.apply_gradients(
            zip(d1_grads, self.d1.trainable_weights)
        )
        d2_grads = d2_tape.gradient(d2_loss, self.d2.trainable_weights)
        self.d2_optimizer.apply_gradients(
            zip(d2_grads, self.d2.trainable_weights)
        )
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_weights)
        )


        # Monitor loss.
        self.gen_loss_tracker.update_state(gen_loss)
        self.d1_loss_tracker.update_state(d1_loss)
        self.d2_loss_tracker.update_state(d2_loss)
        return {
            "generator_loss": self.gen_loss_tracker.result(),
            "d1_loss": self.d1_loss_tracker.result(),
            "d2_loss": self.d2_loss_tracker.result()
        }
    
    def generator_loss(self, inputs, disc_outputs):
        #unpack data
        unsteady, ground_truth = inputs
        d1_out , d2_out = disc_outputs

        #VGG19 SIMILARITY
        features_1 = self.vgg(unsteady)
        features_2 = self.vgg(ground_truth)
        lambda1 = tf.constant([100], dtype = tf.float32)
        features_1 = tf.reshape(features_1,[-1]) #flatten tensor
        features_2 = tf.reshape(features_2,[-1]) #flatten tensor
        #similarity = tf.sqrt(tf.reduce_sum(tf.square(features_1 - features_2))) euclidean distance
        similarity = tf.reduce_mean(tf.abs(features_1 - features_2))    #mean absolute error
        similarity = tf.multiply(similarity,lambda1)
        similarity = tf.reshape(similarity,[1])
        #L1/MAE
        lambda2 = tf.constant([100], dtype = tf.float32)
        l1 =  tf.reduce_mean(tf.abs(unsteady - ground_truth))
        l1 = tf.reshape(l1, [1])
        l1 =  tf.multiply(l1, lambda2)
        #d1 loss 
        d1_loss = tf.reduce_mean(tf.square(tf.ones_like(d1_out) - d1_out))
        d1_loss = tf.reshape(d1_loss,[1])
        #d2 loss
        d2_loss = tf.reduce_mean(tf.square(tf.ones_like(d2_out) - d2_out))
        d2_loss = tf.reshape(d2_loss,[1])
        return similarity + l1 + d1_loss + d2_loss
    
