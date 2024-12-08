import skimage as sk
from skimage.util import img_as_float
import numpy as np
import cv2
from scipy.fft import fft2
from scipy import ndimage
import matplotlib.pyplot as plt

ima = cv2.imread("images/Smeared_LP_Digital_Image_Processing_FA24.png")
center = [658,407]
radius =380
plt.imshow(ima)
plt.show()
skima = sk.transform.warp_polar(ima, center=center, radius=radius, scaling='linear', channel_axis=-1)
plt.imshow(skima)
plt.show()
psf = np.ones((5, 5)) / 25
wienerima1 = sk.restoration.wiener(skima[:,:,0], psf, .2)
wienerima2 = sk.restoration.wiener(skima[:,:,1], psf, .2)
wienerima3 = sk.restoration.wiener(skima[:,:,2], psf, .2)
wienerima = np.stack((wienerima1, wienerima2, wienerima3), axis= 2)
plt.imshow(wienerima)
plt.show()

def polar_linear(imgin, o=None, r=None, output=None, order=1, cont=0):
    img = imgin.transpose()
    if r is None: r = img.shape[0]
    if output is None:
        output = np.zeros((r * 2, r * 2), dtype=img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=img.dtype)
    if o is None: o = np.array(output.shape) / 2 - 0.5
    out_h, out_w = output.shape
    ys, xs = np.mgrid[:out_h, :out_w] - o[:, None, None]
    rs = (ys ** 2 + xs ** 2) ** 0.5
    ts = np.arccos(xs / rs)
    ts[ys < 0] = np.pi * 2 - ts[ys < 0]
    ts *= (img.shape[1] - 1) / (np.pi * 2)
    ndimage.map_coordinates(img, (rs, ts), order=order, output=output)
    return output

restoima = polar_linear(wienerima, r=radius)
plt.imshow(restoima)
plt.show()