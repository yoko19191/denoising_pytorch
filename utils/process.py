import numpy as np

import cv2
from PIL import Image

from skimage.transform import radon, iradon, iradon_sart



def convert_to_opencv_image(image):
    """
    convert PIL.Image object to OpenCV compatible NumPy array
    params:
    - image (PIL.Image): PIL.Image object
    return: 
    - opencv_image (numpy.array): opencv compatible NumPy array
    """
    opencv_image = np.array(image)
    if len(opencv_image.shape) == 3:
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
        
    return opencv_image


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



def add_gaussian_noise(image, mean=0, std_dev=25.0):
    """ add gaussian noise to image, assume image is uint8 range [0, 255]
    params: 
    - image (numpy.array): image to add noise
    - mean of gaussian distribution (float)
    - std_dev: standard deviation of gaussian distribution (float)
    return:
    - noisy image(numpy.array)
    """
    if isinstance(image, Image.Image):
        image = convert_to_opencv_image(image)
    
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_poisson_noise(image, lam = 1.0, scale=1.0):
    """add poisson noise to image, assume image is uint8 range [0, 255]
    params: 
    - image (numpy.array): image to add noise
    - lam: lambda parameter of poisson distribution (float)
    - scale: scale of poisson noise (float)
    return 
    - noisy image (numpy.array)
    """
    if isinstance(image, Image.Image):
        image = convert_to_opencv_image(image)
    
    noise = np.random.poisson(lam, image.shape).astype(np.float32) * scale
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image
