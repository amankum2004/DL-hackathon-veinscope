# # model_loader.py

# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from custom_blocks import EncoderBlock, DecoderBlock

# # Load the model with custom objects
# model = load_model('model_weights.weights.h5', custom_objects={
#     'EncoderBlock': EncoderBlock,
#     'DecoderBlock': DecoderBlock
# })

# # Optionally, compile the model if you plan to use it for training or evaluation
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# model_loader.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from custom_blocks import EncoderBlock, DecoderBlock
import os

def build_model():
    inp = Input(shape=(512, 512, 1))

    p1, c1 = EncoderBlock(32, 2, 0.1, name='ENCODER_1')(inp)
    p2, c2 = EncoderBlock(64, 2, 0.1, name='ENCODER_2')(p1)
    p3, c3 = EncoderBlock(128, 2, 0.2, name='ENCODER_3')(p2)
    p4, c4 = EncoderBlock(256, 2, 0.2, name='ENCODER_4')(p3)

    encoding = EncoderBlock(512, 2, 0.3, pooling=False, name='ENCODING')(p4)

    d1 = DecoderBlock(256, 2, 0.2, name='DECODER_1')([encoding, c4])
    d2 = DecoderBlock(128, 2, 0.2, name='DECODER_2')([d1, c3])
    d3 = DecoderBlock(64, 2, 0.1, name='DECODER_3')([d2, c2])
    d4 = DecoderBlock(32, 2, 0.1, name='DECODER_4')([d3, c1])

    out = Conv2D(1, 1, activation='sigmoid', padding='same')(d4)

    model = Model(inputs=inp, outputs=out)
    return model

# Build the model architecture
model = build_model()

# Load the weights
# model.load_weights('model_weights.weights.h5')
# Dynamically get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model_weights.weights.h5")

# Compile the model (if needed)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
