import cv2
import numpy as np
from gtf_admm_gird import gtf_admm_grid

im = cv2.imread('Lena.png')
im = cv2.resize(im, (52, 52))
im = im.astype(np.float64) / 255

depth = 10
im_3d = np.zeros((im.shape[0], im.shape[1], depth))

for i in range(depth):
    im_3d[:, :, i] = im[:, :, i % 3]

sigma = 0.1
im_3d_noisy = im_3d + sigma * np.random.randn(*im_3d.shape)


lamb = 0.05
rho = lamb
k = 2

x1 = gtf_admm_grid(im_3d_noisy.copy(), k, lamb, rho)

# x2 = gtf_admm_grid_v1(im_3d_noisy.copy(), k, lamb, rho)

