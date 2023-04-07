import os

import numpy as np
from scipy.stats import poisson, norm

import matplotlib.pyplot as plt

import cv2
from PIL import Image

from skimage.transform import radon, iradon, iradon_sart

from torch.utils.data import random_split


## visualization func
def show_images_grid(images, figsize=(10, 10), cmap=None, suptitle=None):
    """show images in a grid
    """
    num_images = len(images)
    num_rows = num_cols = int(np.sqrt(num_images))
    if num_rows * num_cols < num_images:
        num_cols += 1
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(suptitle)
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index >= num_images:
                break
            
            axs[i, j].imshow(images[index], cmap=cmap)
            axs[i, j].axis('off')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

    

def show_error_profile(image_ref, image_target, suptitle=None):
    """Show pair-image and its diff profile."""

    def plot_hist(ax, img, title):
        img_min = img.min()
        img_max = img.max()
        ax.hist(img.ravel(), 256, [img_min, img_max])
        (mu, sigma) = norm.fit(img.ravel())
        ax.set_title(f"{title} (mu:{mu:.3f} sigma:{sigma:.3f})")
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')

    def plot_magnitude_spectrum(ax, img, title):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        im = ax.imshow(magnitude_spectrum, cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(im, ax=ax, label='Magnitude (dB)')

    image_diff = cv2.absdiff(image_ref, image_target)

    # Visualization
    fig, axs = plt.subplots(3, 3, figsize=(30, 20))

    for i, (img, title) in enumerate(zip([image_ref, image_target, image_diff], ["image_ref", "image_target", "error"])):
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].set_title(title)
        plot_hist(axs[1, i], img, f"pixel hist {title}")
        plot_magnitude_spectrum(axs[2, i], img, f"Magnitude Spectrum {title}")
    
    if suptitle is not None:
        fig.suptitle(suptitle)
        
    plt.show()



def show_prepared_image(output_dir, num_image=16, suptitle=None):
    """show prepared images
    """
    filenames = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"filenames len: {len(filenames)}")
    files_path = [os.path.join(output_dir, x) for x in np.random.choice(filenames, num_image)]
    images = [cv2.imread(x, 0) for x in files_path]
    show_images_grid(images, cmap='gray', figsize=(15, 15), suptitle=suptitle)
    
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


## dataset



## projection func
def forward_projection(image, angles=None, circle=True):
    """forward projection of image
    return scaled forward projected sinogram
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if angles is None:
        angles = np.linspace(0., 180., max(image.shape), endpoint=False)
    
    sinogram = radon(image, theta=angles, circle=circle)
    sinogram = ((sinogram - np.min(sinogram)) / (np.max(sinogram) - np.min(sinogram)) * 255).astype(np.uint8)
    return sinogram
    
def filterd_back_projection(sinogram, angles=None, circle=True, filter_name='ramp'):
    """filtered back projection of sinogram
    return scaled fbp recon image
    """
    if isinstance(sinogram, Image.Image):
        sinogram = np.array(sinogram)
        
    if angles is None:
        angles = np.linspace(0., 180., max(sinogram.shape), endpoint=False)
        
    recon = iradon(sinogram, theta=angles, circle=circle, filter_name=filter_name)
    recon = ((recon - np.min(recon)) / (np.max(recon) - np.min(recon)) * 255).astype(np.uint8)
    return recon