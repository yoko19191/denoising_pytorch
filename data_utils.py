import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, iradon_sart
import cv2

## visualization func
def show_images_grid(images, figsize=(10, 10), cmap=None):
    """show images in a grid
    """
    num_images = len(images)
    num_rows = num_cols = int(np.sqrt(num_images))
    if num_rows * num_cols < num_images:
        num_cols += 1
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index >= num_images:
                break
            
            axs[i, j].imshow(images[index], cmap=cmap)
            axs[i, j].axis('off')
            
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    


## experimental func
def add_gaussian_noise(img, mean=0, std_dev=20):
    """ add gaussian noise to image, assume image is uint8
    return (noisy image, noise)
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    """
    noise = np.random.normal(mean, std_dev, img.shape)
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img, noise


def add_poisson_noise(image, lam = 1.0, scale=1.0):
    """add poisson noise to image, assume image is uint8
    return (noisy image, noise)
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
    """
    #image_float = image.astype(np.float32) / 255.0
    noise = np.random.poisson(lam, size=image.shape) * scale
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    #noisy_image = (noisy_image * 255).astype(np.uint8)
    return noisy_image, noise



## projection func
def forward_projection(image, angles=None, circle=True):
    """forward projection of image
    """
    if angles is None:
        angles = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=angles, circle=circle)
    return sinogram
    
