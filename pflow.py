import numpy as np
import scipy
import scipy.ndimage
import scipy.stats as st

sobel_u = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
sobel_v = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])

def gaussian(size):
    lim = size // 2 + (size % 2) / 2
    x = np.linspace(-lim, lim, size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()

def optical_flow(img1, img2, window):
	diff = np.subtract(img1, img2)
	x_grad = scipy.ndimage.filters.convolve(img1, sobel_u, mode='constant')
	y_grad = scipy.ndimage.filters.convolve(img1, sobel_v, mode='constant')
	xx = np.square(x_grad)
	yy = np.square(y_grad)
	xy = np.multiply(x_grad, y_grad)
	xt = np.multiply(x_grad, diff)
	yt = np.multiply(y_grad, diff)
	filt = gaussian(window)
	xx = scipy.ndimage.filters.convolve(xx, filt, mode='constant')
	yy = scipy.ndimage.filters.convolve(yy, filt, mode='constant')
	xy = scipy.ndimage.filters.convolve(xy, filt, mode='constant')
	xt = np.multiply(scipy.ndimage.filters.convolve(xt, filt, mode='constant'), -1.0)
	yt = np.multiply(scipy.ndimage.filters.convolve(yt, filt, mode='constant'), -1.0)
	toReturn = np.zeros((np.shape(img1)[0], np.shape(img1)[1], 2))
	count = 0
	for i in range(np.shape(img1)[0]):
		for j in range(np.shape(img1)[1]):
			h = np.array([[xx[i, j], xy[i, j]], [xy[i, j], yy[i, j]]])
			try:
				h = np.linalg.inv(h)
			except np.linalg.LinAlgError:
				toReturn[i, j, 0] = 0
				toReturn[i, j, 1] = 0
				count += 1
			tderv = np.reshape(np.array([xt[i, j], yt[i, j]]), (2, 1))
			flowv = np.squeeze(np.dot(h, tderv))
			# The new positions of each pixel
			toReturn[i, j, 0] = min(max(i + int(flowv[1]), 0), np.shape(img1)[0] - 1)
			toReturn[i, j, 1] = min(max(j + int(flowv[0]), 0), np.shape(img1)[1] - 1)
	print(count)
	return toReturn
