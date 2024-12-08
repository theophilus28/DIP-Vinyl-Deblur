import cv2 as cv
import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift

def wiener_filter(image, psf, K=0.01):
    image_fft = fft2(image)
    psf_fft = fft2(psf, s=image.shape)
    psf_fft_conj = np.conj(psf_fft)
    wiener_filter = psf_fft_conj / (np.abs(psf_fft)**2 + K)
    result_fft = image_fft * wiener_filter
    result = np.abs(ifft2(result_fft))
    return result