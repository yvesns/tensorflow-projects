import tensorflow as tf
from tensorflow.keras import layers

# This method returns a helper function to compute the binary cross entropy loss.
#
# A binary cross entropy loss is used, among other cases, when there are only two
# possible and exclusive classes, as in the problem of determining if a image is human
# made or not.
#
# A logit is a mapping of a probability [0,1] to [-inf,inf].
# A logit of 0 represents the probability 0.5, and values in the range [-inf,inf] represent
# probabilities below and above 0.5 respectively.
#
# from_logits: Whether to interpret y_pred as a tensor of logit values. By default, we assume
# that y_pred contains probabilities (i.e., values in [0, 1]).
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(1e-4)

def get_optimizer():
    return optimizer

def get_loss(fake_output):
    # fake_output here is the discriminator's evaluations of the fake images.
    # tf.ones_like create a shape filled with ones equal to the input shape.
    # The idea, then, is to compare how much of the images generated were
    # identified as real by the discriminator.
    return cross_entropy(
        tf.ones_like(fake_output), 
        fake_output
    )

# Creates a model to generate data, in this case a image, from a random seed. 
# With training, the model becomes increasingly better at creating seemingly real images.
def make_model():
    # Create a basic sequential model
    model = tf.keras.Sequential()

    # This first dense layer has 7*7*256 units because it will be reshaped and upsampled
    # until the output image shape of (28, 28, 1).
    model.add(
        layers.Dense(
            7*7*256, 
            use_bias = False, 
            input_shape = (100,)
        )
    )

    # Batch normalization applies a transformation that maintains the mean output close to 0 and
    # the output standard deviation close to 1. One of the uses for this is to avoid the shifting
    # of the outputs as the weights change.
    model.add(layers.BatchNormalization())

    # LeakyReLU is used instead of ReLU typically to avoid the problem of dead neurons when the 
    # inputs are mostly negative. Since ReLU returns 0 for any negative input, the weights are 
    # set to 0 when doing backpropagation due to ReLU's derivative.
    model.add(layers.LeakyReLU())

    # A dense layer has a one dimensional output. Thus, this first reshape transforms the output
    # from the first dense layer, which receives the random seed, and creates the base tensor to
    # be transformed into an image.
    model.add(layers.Reshape((7, 7, 256)))

    # Assertion to make sure the shape is correct. In which case it wouldn't?
    # None is the batch size.
    assert model.output_shape == (None, 7, 7, 256)

    # Transposed convolution layer (sometimes called Deconvolution). 128 filters are created. 
    # With strides=(1,1) and padding='same', the filters have the same width and height as the original image.
    model.add(
        layers.Conv2DTranspose(
            128, 
            (5, 5), 
            strides=(1, 1), 
            padding='same', 
            use_bias=False
        )
    )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Now 64 filters are created. Since strides=(2,2), the input matrix is expanded and 
    # the filters' sizes are doubled.
    model.add(
        layers.Conv2DTranspose(
            64, 
            (5, 5), 
            strides=(2, 2), 
            padding='same', 
            use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Only one filter, one more doubling of size.
    model.add(
        layers.Conv2DTranspose(
            1, 
            (5, 5), 
            strides=(2, 2), 
            padding='same', 
            use_bias=False, 
            activation='tanh'
        )
    )
    assert model.output_shape == (None, 28, 28, 1)

    return model

def get_loss(fake_output):
    # fake_output here is the discriminator's evaluations of the fake images.
    # tf.ones_like create a shape filled with ones equal to the input shape.
    # The idea, then, is to compare how much of the images generated were
    # identified as real by the discriminator.
    return cross_entropy(
        tf.ones_like(fake_output), 
        fake_output
    )
