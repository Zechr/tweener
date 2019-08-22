import numpy as np
import scipy.stats as st
from scipy.ndimage.filters import convolve

REGULARIZATION_ALPHA = 0.001
INTENSITY = 0.25
ITERS = 10

kernel_u = np.array([[-1, 1], [-1, 1]]) * INTENSITY
kernel_v = np.array([[-1, -1], [1, 1]]) * INTENSITY
kernel_t = np.ones((2, 2)) * INTENSITY


def gaussian_with_hole(size: int) -> np.ndarray:
    lim = size // 2 + (size % 2) / 2
    x = np.linspace(-lim, lim, size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    kern2d[size // 2, size //2] = 0
    return kern2d / kern2d.sum()


def optical_flow(img1: np.ndarray, img2: np.ndarray, window: int) -> np.ndarray:
    gaussian_filter = gaussian_with_hole(window)

    u_0 = np.zeros([img1.shape[0], img1.shape[1]])
    v_0 = np.zeros([img1.shape[0], img1.shape[1]])

    u_i = u_0
    v_i = v_0

    x_grad = convolve(img1, kernel_u) + convolve(img2, kernel_u)
    y_grad = convolve(img1, kernel_v) + convolve(img2, kernel_v)
    t_grad = convolve(img1, kernel_t) - convolve(img2, kernel_t)

    for _ in range(ITERS):
        u_in_window = convolve(u_i, gaussian_filter)
        v_in_window = convolve(v_i, gaussian_filter)

        gradient = (
            (x_grad * u_in_window + y_grad * v_in_window + t_grad) /
            (REGULARIZATION_ALPHA ** 2 + x_grad ** 2 + y_grad ** 2)
        )

        u_i = u_in_window - x_grad * gradient
        v_i = v_in_window - y_grad * gradient

    row_indices = np.arange(img1.shape[0])[..., np.newaxis]
    col_indices = np.arange(img1.shape[1])[np.newaxis, ...]

    new_rows = np.minimum(np.maximum(v_i + row_indices, 0), img1.shape[0] - 1)
    new_cols = np.minimum(np.maximum(u_i + col_indices, 0), img1.shape[1] - 1)

    coordinates = np.dstack((new_rows, new_cols))

    img1_mask = np.zeros([img1.shape[0], img1.shape[1]])
    img2_mask = np.zeros([img1.shape[0], img1.shape[1]])

    img1_mask[coordinates[:, :, 0], coordinates[:, : 1]] = 1
    vect_dist = np.abs(np.floor(u_i)) + np.abs(np.floor(v_i))
    img2_mask[np.where(vect_dist > 1)] = 1

    half_u = u_i / 2
    half_v = v_i / 2
    half_rows = np.minimum(np.maximum(half_v + row_indices, 0), img1.shape[0] - 1)
    half_cols = np.minimum(np.maximum(half_u + col_indices, 0), img1.shape[1] - 1)
