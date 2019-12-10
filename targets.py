import cv2
import numpy as np
from scipy import signal
import os
import math

def calculate_frame_energy(frame):
	## apply gaussian blur (kernel_size = 3x3)
	blurred = cv2.GaussianBlur(frame, (3, 3), 0).astype(np.float64)

	## calculate the Sobel derivatives
	fx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT_101)
	fy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT_101)
	## energy =  magnitude of the gradient=sqrt(fx^2 + fy^2)
	energy = np.square(fx) + np.square(fy)

	return cv2.normalize(energy, None, 0, 1, cv2.NORM_MINMAX)

def generate_input_channels(current_frame, previous_frame):
	magnitude = np.zeros_like(current_frame).astype(np.float64)
	phase = np.zeros_like(current_frame).astype(np.float64)

	# calculate flow
	if previous_frame is not None:
		flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 5, 30, 3, 10, 1.2, 0)
		magnitude, phase = cv2.cartToPolar(flow[...,0], flow[...,1])

	# calculate grayscale
	current = current_frame.astype(np.float64)
	current = current / 255

	return np.stack([current, magnitude * np.cos(phase), magnitude * np.sin(phase)], axis=-1)

def generate_target(current_frame, current_object_mask):
	energy_map = calculate_frame_energy(current_frame)

	# normalize mask to range [0, 1]
	ret, mask = cv2.threshold(current_object_mask, 0, 255, cv2.THRESH_BINARY)
	mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)

	target = np.maximum(energy_map, mask)

	return target
	
def propagate_importance(frames, targets):
	length = len(frames)

	for count in range(length - 1, 0, -1):
		current_target = targets[count]
		current_frame = frames[count]
		last_target = targets[count - 1]
		last_frame = frames[count - 1]

		flow = cv2.calcOpticalFlowFarneback(last_frame, current_frame, None, 0.5, 5, 30, 3, 10, 1.2, 0)
		displacement_x = flow[...,0]
		displacement_y = flow[...,1]

		for y in range(last_target.shape[0]):
			for x in range(last_target.shape[1]):
				newY = math.floor(y + displacement_y[y, x])
				newX = math.floor(x + displacement_x[y, x])

				if newY >= 0 and newY < current_target.shape[0] and newX >= 0 and newX < current_target.shape[1]:
					last_target[y, x] = max(last_target[y, x], current_target[newY, newX])

		targets[count - 1] = np.expand_dims(last_target, axis=2)

	targets[length - 1] = np.expand_dims(targets[length - 1], axis=2)

	return targets