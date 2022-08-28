import tensorflow as tf
import generator as generator
import discriminator as discriminator
import image_helper
import os
import time
from IPython import display
import matplotlib.pyplot as plt

train_images = None
train_labels = None
train_dataset = None

buffer_size = 60000
batch_size = 256

epochs = 50
noise_dim = 100
num_examples_to_generate = 16

generator_model = generator.make_model()
discriminator_model = discriminator.make_model()

# Reusing the seed overtime allows to visualize the progress more easily in the animated GID
seed = None

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer = generator.get_optimizer(),
    discriminator_optimizer = discriminator.get_optimizer(),
    generator_model = generator_model,
    discriminator_model = discriminator_model
)

def set_buffer_size(size):
    global buffer_size

    buffer_size = size

def set_batch_size(size):
    global batch_size

    batch_size = size

def set_train_images(images):
    global train_images

    train_images = images

def set_train_labels(labels):
    global train_labels

    train_labels = labels

def set_epochs(count):
    global epochs

    epochs = count

def set_noise_dim(dim):
    global noise_dim

    noise_dim = dim
    generate_seed()

def set_num_examples_to_generate(count):
    global num_examples_to_generate

    num_examples_to_generate = count
    generate_seed()

def set_checkpoint_dir(dir):
    global checkpoint_dir

    checkpoint_dir = dir

    update_checkpoint_prefix()

def update_checkpoint_prefix():
    global checkpoint_prefix

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

def generate_train_dataset():
    global train_dataset

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)

def generate_seed():
    global seed

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

def use_mnist_test_dataset():
    global train_images
    global train_labels

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    generate_train_dataset()

def restore_checkpoint():
    checkpoint.restore(
        tf.train.latest_checkpoint(checkpoint_dir)
    )

def start_training():
    train(train_dataset, epochs)

def train(dataset, epochs):
    generate_seed()

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait = True)
        image_helper.generate_and_save_images(generator_model, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    image_helper.generate_and_save_images(generator_model, epochs, seed)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator_model(noise, training = True)
        real_output = discriminator_model(images, training = True)
        fake_output = discriminator_model(generated_images, training = True)

        gen_loss = generator.get_loss(fake_output)
        disc_loss = discriminator.get_loss(real_output, fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    zipped_params = zip(generator_gradients, generator_model.trainable_variables)
    generator.get_optimizer().apply_gradients(zipped_params)

    zipped_params = zip(discriminator_gradients, discriminator_model.trainable_variables)
    discriminator.get_optimizer().apply_gradients(zipped_params)
