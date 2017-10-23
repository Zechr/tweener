import numpy as np
import scipy
import scipy.ndimage

sobel_u = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
sobel_v = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])

def gaussian(size, sigma):
     #m,n = size
     #h, k = m//2, n//2
     #x, y = np.mgrid[-m:m, -n:n]
     #return np.exp(-(x**2 + y**2)/(2*sigma**2))
     return np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])

def optical_flow(img1, img2, window):
	diff = np.subtract(img1, img2)
	x_grad = scipy.ndimage.filters.convolve(img1, sobel_u, mode='constant')
	y_grad = scipy.ndimage.filters.convolve(img1, sobel_v, mode='constant')
	xx = np.square(x_grad)
	yy = np.square(y_grad)
	xy = np.multiply(x_grad, y_grad)
	xt = np.multiply(x_grad, diff)
	yt = np.multiply(y_grad, diff)
	filt = gaussian((window, window), 0.8)
	xx = scipy.ndimage.filters.convolve(xx, filt, mode='constant')
	yy = scipy.ndimage.filters.convolve(yy, filt, mode='constant')
	xy = scipy.ndimage.filters.convolve(xy, filt, mode='constant')
	xt = np.multiply(scipy.ndimage.filters.convolve(xt, filt, mode='constant'), -1.0)
	yt = np.multiply(scipy.ndimage.filters.convolve(yt, filt, mode='constant'), -1.0)
	toReturn = np.zeros((np.shape(img1)[0], np.shape(img1)[1], 2))
	for i in range(np.shape(img1)[0]):
		for j in range(np.shape(img1)[1]):
			if xx[i, j]*yy[i, j] - xy[i, j]**2 == 0:
				toReturn[i, j, 0] = 0
				toReturn[i, j, 1] = 0
				continue
			h = np.array([[xx[i, j], xy[i, j]], [xy[i, j], yy[i, j]]])
			h = np.linalg.inv(h)
			tderv = np.reshape(np.array([xt[i, j], yt[i, j]]), (2, 1))
			flowv = np.squeeze(np.dot(h, tderv))
			toReturn[i, j, 0] = int(flowv[0])
			toReturn[i, j, 1] = int(flowv[1])
	return toReturn

def flow_conversion(img, flow):
	toReturn = np.zeros((np.shape(img)[0], np.shape(img)[1]))
	for i in range(np.shape(img)[0]):
		for j in range(np.shape(img)[1]):
			newi = min(max(i + int(flow[i, j, 1]), 0), np.shape(img)[0] - 1)
			newj = min(max(j + int(flow[i, j, 0]), 0), np.shape(img)[1] - 1)
			toReturn[i, j] = img[newi, newj]
	return toReturn