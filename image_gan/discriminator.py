import tensorflow as tf
from tensorflow.keras import layers

# See generator.py for more general information.

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(1e-4)

def get_optimizer():
    return optimizer

def get_loss(real_output, fake_output):
    # fake_output here is the discriminator's evaluations of the real images.
    real_loss = cross_entropy(
        tf.ones_like(real_output), 
        real_output
    )

    # fake_output here is the discriminator's evaluations of the fake images.
    fake_loss = cross_entropy(
        tf.zeros_like(fake_output), 
        fake_output
    )

    total_loss = real_loss + fake_loss

    return total_loss

def make_model():
    model = tf.keras.Sequential()

    # Since strides=(2,2) and padding='same', the final filter size will be half the input
    # size. 64 filters of size 14x14 are produced.
    model.add(
        layers.Conv2D(
            64, 
            (5, 5), 
            strides=(2, 2), 
            padding='same',
            input_shape=[28, 28, 1]
        )
    )

    # Avoid dead neurons by considering to some extent negative inputs.
    model.add(layers.LeakyReLU())

    # Randomly deactivate certain neurons at a 30% rate to improve the individual quality
    # of each neuron.
    model.add(layers.Dropout(0.3))

    # Now, produce 128 filters of size 7x7.
    model.add(
        layers.Conv2D(
            128, 
            (5, 5), 
            strides=(2, 2), 
            padding='same'
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Produces values in a single dimension. The total quantity of values will be 7 x 7 x 128.
    model.add(layers.Flatten())

    # Produces a single value in a single dimension. The model will be trained to output 
    # positive values for real images, and negative values for fake images.
    model.add(layers.Dense(1))

    return model
