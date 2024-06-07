# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 08:01:36 2020

@author: Ardhendu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:33:32 2019

@author: Ardhendu
"""
import tensorflow as tf
from SpectralNormalizationKeras import ConvSN2DTranspose
from tensorflow.keras import layers
def hw_flatten(x):
    x_shape = tf.shape(x)
    return tf.reshape(x, [x_shape[0], -1, x_shape[-1]]) # return [BATCH, W*H, CHANNELS]

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        self.data_format = tf.keras.backend.image_data_format()
        assert self.data_format in {'channels_last', 'channels_first'}, 'data_format must be in {channels_last, channels_first}'
        self.filters = filters
        super(SelfAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.gamma = self.add_weight(
        shape=(1,), name="gamma", initializer="zeros", trainable=True)
        self.proj_o = layers.Conv2D(
        filters=input_shape[-1],  # Specify filters as an integer
        kernel_size=(1, 1), 
        activation=None, 
        padding="same")
        super(SelfAttention, self).build(input_shape)
        
    def call(self, x):
        img = x
        f = ConvSN2DTranspose(self.filters // 8, (1, 1), padding="same")(img)
        g = ConvSN2DTranspose(self.filters // 8, (1, 1), padding="same")(img)
        h = ConvSN2DTranspose(self.filters // 8, (1, 1), padding="same")(img)
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  
        beta = tf.nn.softmax(s)  
        o = tf.matmul(beta, hw_flatten(h))  
        o = tf.reshape(
            o,
            shape=[tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2], self.filters // 8],
        ) 
        gamma = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.gamma, 1), 1), 1)
        o = self.proj_o(o)         
        img = gamma * o + img  
        return img
    
    def compute_output_shape(self, input_shape):
        return input_shape  
    
    def get_config(self):
        config = {'filters': self.filters}
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
