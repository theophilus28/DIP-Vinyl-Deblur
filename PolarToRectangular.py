import skimage as sk
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

def estimate_psf(length, angle):
    psf = np.zeros((length, length))
    center = length // 2
    for i in range(length):
        x = center + int((i - center) * np.cos(angle))
        y = center + int((i - center) * np.sin(angle))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1
    return psf / psf.sum()

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

ima = cv2.imread("images/IMG_20241208_182953515.jpg")
ima = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)

#1/8th second 
#center = [1635,1950]
#radius =1000
#1/10th second
center = [1550,1865]
radius =1010
#1/6th second
#center = [1585,1830]
#radius =1038
#1/4th second
#center = [1500,1870]
#radius =1010
plt.imshow(ima)
plt.show()
skima = sk.transform.warp_polar(ima, center=center, radius=radius, scaling='linear', channel_axis=-1)
plt.imshow(skima)
plt.show()

#vinyl spins at 33.33 rotations per minute. psf length = 33.3/60 seconds * shutter speed(in seconds) *360 degrees
#1/8th=25, 1/6th=33.33(34), 1/4th=50(49), 1/10th=20
psf = estimate_psf(20, np.pi *.5)
wienerima1 = sk.restoration.unsupervised_wiener(skima[:,:,0], psf)
wienerima2 = sk.restoration.unsupervised_wiener(skima[:,:,1], psf)
wienerima3 = sk.restoration.unsupervised_wiener(skima[:,:,2], psf)


restoima1 = polar_linear(wienerima1[0], r=radius)
restoima2 = polar_linear(wienerima2[0], r=radius)
restoima3 = polar_linear(wienerima3[0], r=radius)
restoima = np.stack((restoima1,restoima2, restoima3), axis=2)
plt.imshow(restoima)
plt.show()