# from tensorflow.keras.layers import (
#     Input, Conv2D, UpSampling2D, Dropout, MaxPooling2D,
#     Layer, concatenate, BatchNormalization
# )

# # Encoding block for U-net architecture

# class EncoderBlock(Layer):
#     def __init__(self, filters, kernel_size, rate, pooling = True, **kwargs):
#         super(EncoderBlock, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.rate = rate
#         self.pooling = pooling
        
#         self.conv1 = Conv2D(filters, kernel_size, activation = 'relu', strides = 1,  padding = 'same' ,kernel_initializer = 'he_normal')
#         self.conv2 = Conv2D(filters, kernel_size, activation= 'relu', strides = 1, padding = 'same', kernel_initializer = 'he_normal') 
#         self.drop = Dropout(rate)
#         self.pool = MaxPooling2D()
    
    
#     def call(self, inputs):
#         X = self.conv1(inputs)
#         X = self.drop(X)
#         X = self.conv2(X)
#         if self.pooling:
#             P = self.pool(X)
#             return P,X
#         else:
#             return X
     
    
#     def get_config(self):
#         base_config = super().get_config()
#         return {
#             **base_config,
#             "filters" : self.filters,
#             "kernel_size": self.kernel_size,
#             "rate" : self.rate,
#             "pooling" : self.pooling }
        
        
# # Decoding block for U-net architecture

# class DecoderBlock(Layer):
#     def __init__(self, filters, kernel_size, rate, **kwargs):
#         super(DecoderBlock, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.rate = rate
        
#         self.up = UpSampling2D()
#         self.nn = EncoderBlock(filters, kernel_size, rate, pooling = False)
    
    
#     def call(self, inputs):
#         inputs, skip_inputs = inputs
#         X = self.up(inputs)
#         C = concatenate([X, skip_inputs ])
#         X = self.nn(C)
#         return X
    
    
#     # def get_config(self):
#     #     base_config = super().get_config()
#     #     return {
#     #         **base_config,
#     #         "filters" : self.filters,
#     #         "kernel_size": self.kernel_size,
#     #         "rate" : self.rate,
#     #         "pooling" : self.pooling }
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "filters": self.filters,
#             "kernel_size": self.kernel_size,
#             "rate": self.rate
#         })
#         return config

        
        


import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, UpSampling2D, Dropout, MaxPooling2D,
    Layer, concatenate
)

class EncoderBlock(Layer):
    def __init__(self, filters, kernel_size, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.rate = rate
        self.pooling = pooling

        self.conv1 = Conv2D(filters, kernel_size, activation='relu', strides=1,
                            padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(filters, kernel_size, activation='relu', strides=1,
                            padding='same', kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        if self.pooling:
            self.pool = MaxPooling2D()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.drop(x)
        x = self.conv2(x)
        if self.pooling:
            p = self.pool(x)
            return p, x
        else:
            return x

    def get_config(self):
        config = super(EncoderBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'rate': self.rate,
            'pooling': self.pooling
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DecoderBlock(Layer):
    def __init__(self, filters, kernel_size, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.rate = rate

        self.up = UpSampling2D()
        self.nn = EncoderBlock(filters, kernel_size, rate, pooling=False)

    def call(self, inputs):
        inputs, skip_inputs = inputs
        x = self.up(inputs)
        x = concatenate([x, skip_inputs])
        x = self.nn(x)
        return x

    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'rate': self.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
