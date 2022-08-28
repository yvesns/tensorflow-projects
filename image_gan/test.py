import image_gan as image_gan

MNIST, CELEB = range(2)

checkpoint_paths = {
    MNIST: './mnist_checkpoints',
    CELEB: './celeb_checkpoints',
}

image_gan.set_checkpoint_dir(checkpoint_paths[MNIST])
image_gan.use_mnist_test_dataset()
image_gan.restore_checkpoint()
image_gan.start_training()
