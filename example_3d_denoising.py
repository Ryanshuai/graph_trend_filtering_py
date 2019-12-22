import cv2
import numpy as np
from gtf_admm_gird import gtf_admm_grid
from gtf_admm_gird_v1 import gtf_admm_grid_v1

im = cv2.imread('Lena.bmp')
# im = cv2.resize(im, (6, 6))
im = im[0:6, 0:6, :]
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

im = im.astype(np.float64) / 255

im_3d = im

# depth = 10
# im_3d = np.zeros((im.shape[0], im.shape[1], depth))
#
# for i in range(depth):
#     im_3d[:, :, i] = im[:, :, i % 3]

# sigma = 0.1
sigma = 0.0
im_3d_noisy = im_3d + sigma * np.random.randn(*im_3d.shape)

lamb = 0.05
rho = lamb
k = 2

# x1 = gtf_admm_grid(im_3d_noisy.copy(), k, lamb, rho)
# x1 = np.clip(x1, 0, 1.)
# x1 = (x1*255).astype(np.uint8)
# cv2.imwrite('x1_1_3.png', x1[:, :, 0:3])

x2 = gtf_admm_grid_v1(im_3d_noisy.copy(), k, lamb, rho)
