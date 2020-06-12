
import cv2
import matplotlib
import numpy as np
import random


def dataAugmentation(x, dataAugmentationRate=0.5, verbose=False):
    # This function present several DA methods. Some methods might requires adapting the labels as well.

	if (random.random() < dataAugmentationRate): return x
	if (verbose): matplotlib.image.imsave("./temp/DA_0_Original.png", x)

	if (random.random() < 0.5):
		# Horizontal Flip
		x = np.flip(x, 1) # Flip along axe 1 -> Horizontal flip. Axe 0 -> Vectical flip.
		matplotlib.image.imsave("./temp/DA_1_Flipped.png", x)

	# Gaussian Blur
	x = cv2.GaussianBlur(x, (5, 5), 0)
	if (verbose): matplotlib.image.imsave("./temp/DA_2_Blured.png", x)

	# Colored Saturation
	n = np.random.normal(0, 0.05, x.shape)
	x = np.minimum(np.maximum(x + n, 0),1)
	if (verbose): matplotlib.image.imsave("./temp/DA_3_ColorSaturation.png", x)

	# Brightness Offset
	r = random.random() * 0.05
	x = np.where((1 - x) < r, 1, x + r)
	if (verbose): matplotlib.image.imsave("./temp/DA_4_Brightness.png", x)

	# Salt&Pepper Noise
	s_p = np.random.randint(100 + 1, size=x.shape) - 100//2
	s_p = np.where(abs(s_p) < 100//2, 0, s_p)
	s_p = np.where(s_p == 0, 0, 1 * np.sign(s_p))
	s_p[1] = s_p[0]
	s_p[2] = s_p[0]
	x = np.maximum(0,np.minimum(1, x + s_p))
	if (verbose): matplotlib.image.imsave("./temp/DA_5_Salt&Pepper.png", x)

	# Width/Height Shift (source : https://medium.com/@schatty/image-augmentation-in-numpy-the-spell-is-simple-but-quite-unbreakable-e1af57bb50fd )
	def translate(img, shift=10, direction='right', roll=True):
		assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
		img = img.copy()
		if direction == 'right':
			right_slice = img[:, -shift:].copy()
			img[:, shift:] = img[:, :-shift]
			if roll:
				img[:,:shift] = np.fliplr(right_slice)
		if direction == 'left':
			left_slice = img[:, :shift].copy()
			img[:, :-shift] = img[:, shift:]
			if roll:
				img[:, -shift:] = left_slice
		if direction == 'down':
			down_slice = img[-shift:, :].copy()
			img[shift:, :] = img[:-shift,:]
			if roll:
				img[:shift, :] = down_slice
		if direction == 'up':
			upper_slice = img[:shift, :].copy()
			img[:-shift, :] = img[shift:, :]
			if roll:
				img[-shift:,:] = upper_slice
		return img

	horizontalDirection = "left" if (random.random() < 0.5) else "right"
	horizontalAmplitude = 1 + int(random.random() * 15) # Hard-coded 15 pixels max
	verticalDirection = "up" if (random.random() < 0.5) else "down"
	verticalAmplitude = 1 + int(random.random() * 15) # Hard-coded 15 pixels max

	x = translate(x, shift=horizontalAmplitude, direction=horizontalDirection)
	x = translate(x, shift=verticalAmplitude, direction=verticalDirection)
	if (verbose): matplotlib.image.imsave("./temp/DA_6_Shift.png", x)

	# Rotation (source : https://medium.com/@schatty/image-augmentation-in-numpy-the-spell-is-simple-but-quite-unbreakable-e1af57bb50fd )
	from scipy.ndimage import rotate
	def rotate_img(img, angle, bg_patch=(5,5)):
		assert len(img.shape) <= 3, "Incorrect image shape"
		rgb = len(img.shape) == 3
		if rgb:
			bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
		else:
			bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
		img = rotate(img, angle, reshape=False)
		mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
		img[mask] = bg_color
		return img

	angle = int(random.random() * 20 - 10) # Hard-coded 10 degrees max
	x = rotate_img(x, angle)
	if (verbose): matplotlib.image.imsave("./temp/DA_7_Rotation.png", x)
	return x
