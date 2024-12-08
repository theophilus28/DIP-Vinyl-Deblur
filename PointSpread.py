import cv2
import numpy as np
from scipy.signal import convolve2d
from WeinerFilter import wiener_filter
import PIL
from PIL import Image
import matplotlib.pyplot as plt


def estimate_psf(length, angle):
    """
    Estimate the Point Spread Function (PSF) for motion blur.
    """
    psf = np.zeros((length, length))
    center = length // 2
    for i in range(length):
        x = center + int((i - center) * np.cos(angle))
        y = center + int((i - center) * np.sin(angle))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1
    return psf / psf.sum()

def deblur_image(image, psf, iterations=30):
    """
    Deblur image using the Wiener filter with a known PSF.
    """
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf, s=image.shape)
    psf_fft_conj = np.conj(psf_fft)
    epsilon = 1e-3

    deblurred_fft = (psf_fft_conj / (psf_fft * psf_fft_conj + epsilon)) * image_fft
    deblurred = np.abs(np.fft.ifft2(deblurred_fft))
    return np.clip(deblurred, 0, 255).astype(np.uint8)

def main(input_image_path, output_image_path, psf_length=15, psf_angle=0.0):
    """
    Main function to read, deblur, and save the image.
    """
    # Load the blurred image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not load the image. Check the file path.")

    # Estimate PSF for motion blur
    psf = estimate_psf(psf_length, psf_angle)

    # Deblur the image
    deblurred_image = wiener_filter(image, psf)

    # Save the deblurred image
    cv2.imwrite(output_image_path, deblurred_image)
    print(f"Deblurred image saved to {output_image_path}")
    img = PIL.Image.open(output_image_path)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    # Set file paths
    input_path = "images/IMG_20241208_124234494_BURST000_COVER.jpg"  # Input image path
    #input_path = "images/ultraBlurred.jpg"  # Input image path
    output_path = "output/deblurred_vinyl.jpg"  # Output image path

    # Customize PSF parameters
    psf_length = 15  # Length of blur (approximate radius of the spinning vinyl blur)
    psf_angle = -np.pi / 2  # Direction of motion (vertical blur)

    main(input_path, output_path, psf_length, psf_angle)
