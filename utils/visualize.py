import os

import numpy as np
from scipy.stats import poisson, norm

import matplotlib.pyplot as plt

import cv2
from PIL import Image


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
    



