import skimage as sk
import numpy as np
import cv2
from scipy.fft import fft2
from scipy import ndimage
import matplotlib.pyplot as plt

ima = cv2.imread("images/Smeared_LP_Digital_Image_Processing_FA24.png")
center = [407,658]
radius =380
ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
skima = sk.transform.warp_polar(ima, center=center, radius=radius)
fourskima = fft2(skima)
wienerima = sk.restoration.unsupervised_wiener(skima, fourskima)


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

restoima = polar_linear(wienerima[0], r=radius)
plt.imshow(restoima)
plt.show()