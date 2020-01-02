import cv2
import numpy as np
from scipy import signal
import os
import math
import random

## calculates the energy of the frame(grayscale) using Sobel filter
def calculate_frame_energy(frame):
	## apply gaussian blur (kernel_size = 3x3)
	blurred = cv2.GaussianBlur(frame, (3, 3), 0).astype(np.float64)

	## calculate the Sobel derivatives
	fx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT_101)
	fy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT_101)
	## energy =  magnitude of the gradient=sqrt(fx^2 + fy^2)
	energy = np.square(fx) + np.square(fy)

	return cv2.normalize(energy, None, 0, 1, cv2.NORM_MINMAX)

## generates the input image for the current_frame(grayscale): 
## grayscale, x-component of flow from previous_frame, y-component of flow from previous_frame(grayscale)
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

## generates saliency map for a single frame(current_frame in grayscale), 
## using the segmentation mask (current_object_mask in grayscale)
def generate_target(current_frame, current_object_mask):
	energy_map = calculate_frame_energy(current_frame)

	# normalize mask to range [0, 1]
	ret, mask = cv2.threshold(current_object_mask, 0, 255, cv2.THRESH_BINARY)
	mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)

	target = np.maximum(energy_map, mask)

	return target

## propagates the saliency map of the nth frame back to the (n-1)th frame.
## The propagation starts from the last frame of the video and ends at the first frame.
## frames is the list of frames in the video
## targets is the list of saliency maps corresponding to each frame created using generate_target()	
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

## generates a list of inputs and corresponding saliency maps (targets) to be fed to the network
def generate_inputs_and_targets(frames, masks, generate_targets=False):
	## reverse the list of frames and their segmentation masks
	## the last frame is processed first and the saliency map is propagated backwards
	frames = list(reversed(frames))
	masks = list(reversed(masks))

	## list of inputs and corresponding saliency maps (targets)
	## each input has 3 channels; grayscale of the current frame, x-component of flow to the next frame, y-component of the next frame
	inputs = []
	targets = []

	next_frame = None
	next_target = None
	target = None

	for frame, mask in zip(frames, masks):
		displacement_x = np.zeros_like(frame).astype(np.float64)
		displacement_y = np.zeros_like(frame).astype(np.float64)

		if generate_targets:
			## generate saliency map for the nth frame (frame)
			target = generate_target(frame, mask)

		## calculate flow
		if next_frame is not None:
			flow = cv2.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 3, 3, 5, 1.1, 0)
			displacement_x = flow[...,0]
			displacement_y = flow[...,1]

			if generate_targets:
				## propagate the saliency map of the nth frame (frame) to the saliency map of the (n-1)th frame (next_frame)
				for y in range(target.shape[0]):
					for x in range(target.shape[1]):
						newY = math.floor(y + displacement_y[y, x])
						newX = math.floor(x + displacement_x[y, x])

						if newY >= 0 and newY < next_target.shape[0] and newX >= 0 and newX < next_target.shape[1]:
							target[y, x] = max(target[y, x], next_target[newY, newX])

		## calculate grayscale
		current = frame.astype(np.float64)
		current = current / 255	

		## input: 3 channels; grayscale, x-component of flow to the next frame, y-component of the next frame
		inputs.append(np.stack([current, displacement_x, displacement_y], axis=-1))
		
		if generate_targets:
			targets.append(np.expand_dims(target, axis=2))
			next_target = target

		next_frame = frame

	return inputs, targets

## generates a list of colour frames (images) and corresponding saliency maps (targets)
def generate_full_images_and_targets(images, frames, masks):
	## reverse the list of frames and their segmentation masks
	## the last frame is processed first and the saliency map is propagated backwards
	images = list(reversed(images))
	frames = list(reversed(frames))
	masks = list(reversed(masks))

	## list of inputs and corresponding saliency maps (targets)
	## each input has 3 channels; grayscale of the current frame, x-component of flow to the next frame, y-component of the next frame
	targets = []

	next_frame = None
	next_target = None
	target = None

	for image, frame, mask in zip(images, frames, masks):
		displacement_x = np.zeros_like(frame).astype(np.float64)
		displacement_y = np.zeros_like(frame).astype(np.float64)

		## generate saliency map for the nth frame (frame)
		target = generate_target(frame, mask)

		## calculate flow
		if next_frame is not None:
			flow = cv2.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 3, 3, 5, 1.1, 0)
			displacement_x = flow[...,0]
			displacement_y = flow[...,1]

			## propagate the saliency map of the nth frame (frame) to the saliency map of the (n-1)th frame (next_frame)
			for y in range(target.shape[0]):
				for x in range(target.shape[1]):
					newY = math.floor(y + displacement_y[y, x])
					newX = math.floor(x + displacement_x[y, x])

					if newY >= 0 and newY < next_target.shape[0] and newX >= 0 and newX < next_target.shape[1]:
						target[y, x] = max(target[y, x], next_target[newY, newX])
		
		targets.append(np.expand_dims(target, axis=2))
		next_target = target

		next_frame = frame

	return images, targets

## saves colour images and corresponding saliency map for 'video' in image_folder and target_folder respectively
def save_full_sequence(in_folder, mask_folder, video, image_folder, target_folder):
	images = []
	frames = []
	masks = []

	video_path = os.path.join(in_folder, video)
	if os.path.isdir(video_path):
		mask_folder_path = os.path.join(mask_folder, video)
		target_folder_path = os.path.join(target_folder, video)
		image_folder_path = os.path.join(image_folder, video)
					
		frame_count = len(os.listdir(video_path))
		for count in range(frame_count):
			frame_path = os.path.join(video_path, '%05d.jpg'%count)
			mask_path = os.path.join(mask_folder_path, '%05d.png'%count)

			image = cv2.imread(frame_path)
			frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			mask = cv2.imread(mask_path, 0)

			images.append(image)
			frames.append(frame)
			masks.append(mask)

		images, targets = generate_full_images_and_targets(images, frames, masks)

		if not os.path.isdir(target_folder_path):
			os.mkdir(target_folder_path)

		if not os.path.isdir(image_folder_path):
			os.mkdir(image_folder_path)
		
		count = 0
		for image, target in zip(images, targets):
			new_target_path = os.path.join(target_folder_path, '%05d.npy'%(frame_count - count - 1))
			np.save(new_target_path, target)

			new_image_path = os.path.join(image_folder_path, '%05d.png'%(frame_count - count - 1))
			cv2.imwrite(new_image_path, image)

			print(new_image_path)

			count += 1

## generates list of temporal slices of input and corresponding saliency map (target)
## the input_slices and target_slices are concatenated along the channels axis
## Each slice is of 'slice_size' length, padded if required
def generate_inputs_and_targets_for_temporal_slice(frames, masks, slice_size=1, generate_targets=False):
	## reverse the list of frames and their segmentation masks
	## the last frame is processed first and the saliency map is propagated backwards
	frames = list(reversed(frames))
	masks = list(reversed(masks))

	## list of inputs and corresponding saliency map (targets)
	## each input has 3 channels; grayscale of the current frame, x-component of flow to the next frame, y-component of the next frame
	inputs = []
	targets = []
	input_slices = []
	propagated_targets = []

	next_frame = None
	target = None

	for frame, mask in zip(frames, masks):
		displacement_x = np.zeros_like(frame).astype(np.float64)
		displacement_y = np.zeros_like(frame).astype(np.float64)

		if generate_targets:
			## generate saliency map for the nth frame (frame)
			target = generate_target(frame, mask)

		## calculate flow
		if next_frame is not None:
			flow = cv2.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 3, 3, 5, 1.1, 0)
			displacement_x = flow[...,0]
			displacement_y = flow[...,1]		

		## calculate grayscale
		current = frame.astype(np.float64)
		current = current / 255	

		## input: 3 channels; grayscale, x-component of flow to the next frame, y-component of the next frame
		inputs.append(np.stack([current, displacement_x, displacement_y], axis=2))
		
		if generate_targets:
			targets.append(target)

		next_frame = frame

	frame_count = len(inputs)
	## propagate within temporal slice
	if generate_targets:
		inputs_slice = []
		targets_slice = []
		for count in range(frame_count):
			if count >= slice_size - 1:
				inputs_slice = inputs[count - slice_size + 1:count + 1]
				targets_slice = [np.copy(target) for target in targets[count - slice_size + 1:count + 1]]
			else:
				inputs_slice = inputs[:count + 1]
				targets_slice = [np.copy(target) for target in targets[:count + 1]]
			
			index = 0
			next_target = None
			for image, target in zip(inputs_slice, targets_slice):			
				if next_target is not None:
					displacement_x = image[:, :, 1]
					displacement_y = image[:, :, 2]
					## propagate the saliency map of the nth frame (frame) to the saliency map of the (n-1)th frame (next_frame)
					for y in range(target.shape[0]):
						for x in range(target.shape[1]):
							newY = math.floor(y + displacement_y[y, x])
							newX = math.floor(x + displacement_x[y, x])

							if newY >= 0 and newY < next_target.shape[0] and newX >= 0 and newX < next_target.shape[1]:
								target[y, x] = max(target[y, x], next_target[newY, newX])

				next_target = target
				targets_slice[index] = target

				index += 1
			
			slice_length = len(targets_slice)
			if slice_length < slice_size:
				padded_targets = [np.zeros_like(targets_slice[-1])] * (slice_size - slice_length)
				padded_targets.extend(targets_slice)
				propagated_targets.append(np.stack(padded_targets, axis=2))
			else:
				propagated_targets.append(np.stack(targets_slice, axis=2))

	inputs_slice = []
	for count in range(frame_count):
		if count >= slice_size - 1:
			inputs_slice = inputs[count - slice_size + 1:count + 1]
		else:
			inputs_slice = inputs[:count + 1]

		slice_length = len(inputs_slice)
		if slice_length < slice_size:
			padded_inputs = [np.zeros_like(inputs_slice[-1])] * (slice_size - slice_length)
			padded_inputs.extend(inputs_slice)
			input_slices.append(np.concatenate(padded_inputs, axis=2))
		else:
			input_slices.append(np.concatenate(inputs_slice, axis=2))

	return input_slices, propagated_targets

## generates list of temporal slices of input and corresponding saliency map (target)
## Since these are passed to an rnn, the input_slices and target_slices are not concatenated along the channels axis
## Each slice is of at most 'slice_size' length, not padded
def generate_inputs_and_targets_for_temporal_slice_for_rnn(frames, masks, slice_size=1, generate_targets=False):
	## reverse the list of frames and their segmentation masks
	## the last frame is processed first and the saliency map is propagated backwards
	frames = list(reversed(frames))
	masks = list(reversed(masks))

	## list of inputs and corresponding saliency map (targets)
	## each input has 3 channels; grayscale of the current frame, x-component of flow to the next frame, y-component of the next frame
	inputs = []
	targets = []
	input_slices = []
	propagated_targets = []

	next_frame = None
	target = None

	for frame, mask in zip(frames, masks):
		displacement_x = np.zeros_like(frame).astype(np.float64)
		displacement_y = np.zeros_like(frame).astype(np.float64)

		if generate_targets:
			## generate saliency map for the nth frame (frame)
			target = generate_target(frame, mask)

		## calculate flow
		if next_frame is not None:
			flow = cv2.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 3, 3, 5, 1.1, 0)
			displacement_x = flow[...,0]
			displacement_y = flow[...,1]		

		## calculate grayscale
		current = frame.astype(np.float64)
		current = current / 255	

		## input: 3 channels; grayscale, x-component of flow to the next frame, y-component of the next frame
		inputs.append(np.stack([current, displacement_x, displacement_y], axis=2))
		
		if generate_targets:
			targets.append(target)

		next_frame = frame

	frame_count = len(inputs)
	## propagate within temporal slice
	if generate_targets:
		inputs_slice = []
		targets_slice = []
		for count in range(frame_count):
			if count >= slice_size - 1:
				inputs_slice = inputs[count - slice_size + 1:count + 1]
				targets_slice = [np.copy(target) for target in targets[count - slice_size + 1:count + 1]]
			else:
				inputs_slice = inputs[:count + 1]
				targets_slice = [np.copy(target) for target in targets[:count + 1]]
			
			index = 0
			next_target = None
			for image, target in zip(inputs_slice, targets_slice):			
				if next_target is not None:
					displacement_x = image[:, :, 1]
					displacement_y = image[:, :, 2]
					## propagate the saliency map of the nth frame (frame) to the saliency map of the (n-1)th frame (next_frame)
					for y in range(target.shape[0]):
						for x in range(target.shape[1]):
							newY = math.floor(y + displacement_y[y, x])
							newX = math.floor(x + displacement_x[y, x])

							if newY >= 0 and newY < next_target.shape[0] and newX >= 0 and newX < next_target.shape[1]:
								target[y, x] = max(target[y, x], next_target[newY, newX])

				next_target = target
				targets_slice[index] = np.expand_dims(target, axis=2)

				index += 1
			
			propagated_targets.append(np.array(targets_slice))

	inputs_slice = []
	for count in range(frame_count):
		if count >= slice_size - 1:
			inputs_slice = inputs[count - slice_size + 1:count + 1]
		else:
			inputs_slice = inputs[:count + 1]

		input_slices.append(np.array(inputs_slice))

	return input_slices, propagated_targets

## generates a list of temporal slices of colour frames (images) and corresponding saliency maps (targets)
## Each slice is of 'slice_size' length, padded if required
def generate_images_and_targets(images, frames, masks, slice_size=1):
	## reverse the list of frames and their segmentation masks
	## the last frame is processed first and the saliency map is propagated backwards
	images = list(reversed(images))
	frames = list(reversed(frames))
	masks = list(reversed(masks))

	## list of inputs and corresponding saliency map (targets)
	## each input has 3 channels; grayscale of the current frame, x-component of flow to the next frame, y-component of the next frame
	targets = []
	image_slices = []
	propagated_targets = []

	next_frame = None
	target = None

	for frame, mask in zip(frames, masks):
		## generate saliency map for the nth frame (frame)
		target = generate_target(frame, mask)
		targets.append(target)

	frame_count = len(frames)
	## propagate within temporal slice
	inputs_slice = []
	targets_slice = []
	for count in range(frame_count):
		if count >= slice_size - 1:
			images_slice = images[count - slice_size + 1:count + 1]
			inputs_slice = frames[count - slice_size + 1:count + 1]
			targets_slice = [np.copy(target) for target in targets[count - slice_size + 1:count + 1]]
		else:
			images_slice = images[:count + 1]
			inputs_slice = frames[:count + 1]
			targets_slice = [np.copy(target) for target in targets[:count + 1]]
		
		index = 0
		next_target = None
		next_frame = None
		for frame, target in zip(inputs_slice, targets_slice):			
			if next_frame is not None:
				flow = cv2.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 3, 3, 5, 1.1, 0)
				displacement_x = flow[...,0]
				displacement_y = flow[...,1]
				## propagate the saliency map of the nth frame (frame) to the saliency map of the (n-1)th frame (next_frame)
				for y in range(target.shape[0]):
					for x in range(target.shape[1]):
						newY = math.floor(y + displacement_y[y, x])
						newX = math.floor(x + displacement_x[y, x])

						if newY >= 0 and newY < next_target.shape[0] and newX >= 0 and newX < next_target.shape[1]:
							target[y, x] = max(target[y, x], next_target[newY, newX])

			next_frame = frame
			next_target = target
			targets_slice[index] = np.expand_dims(target, axis=2)

			index += 1
		
		slice_length = len(targets_slice)
		if slice_length < slice_size:
			padded_targets = [np.zeros_like(targets_slice[-1])] * (slice_size - slice_length)
			padded_targets.extend(targets_slice)
			propagated_targets.append(padded_targets)

			padded_inputs = [np.zeros_like(images_slice[-1])] * (slice_size - slice_length)
			padded_inputs.extend(images_slice)
			image_slices.append(padded_inputs)
		else:
			propagated_targets.append(targets_slice)
			image_slices.append(images_slice)

	return image_slices, propagated_targets

## saves temporal slices of colour images and corresponding saliency map for 'video' in image_folder and target_folder respectively
## Each slice is of 'slice_size' length, padded if required
def save_temporal_slices(in_folder, mask_folder, video, image_folder, target_folder, slice_size):
	images = []
	frames = []
	masks = []

	video_path = os.path.join(in_folder, video)
	if os.path.isdir(video_path):
		mask_folder_path = os.path.join(mask_folder, video)
		target_folder_path = os.path.join(target_folder, video)
		image_folder_path = os.path.join(image_folder, video)
					
		frame_count = len(os.listdir(video_path))
		for count in range(frame_count):
			frame_path = os.path.join(video_path, '%05d.jpg'%count)
			mask_path = os.path.join(mask_folder_path, '%05d.png'%count)

			image = cv2.imread(frame_path)
			frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			mask = cv2.imread(mask_path, 0)

			images.append(image)
			frames.append(frame)
			masks.append(mask)

		image_slices, target_slices = generate_images_and_targets(images, frames, masks, slice_size)

		if not os.path.isdir(target_folder_path):
			os.mkdir(target_folder_path)

		if not os.path.isdir(image_folder_path):
			os.mkdir(image_folder_path)
		
		for count in range(frame_count):
			target_frame_path = os.path.join(target_folder_path, '%05d'%(frame_count - count - 1))
			if not os.path.isdir(target_frame_path):
				os.mkdir(target_frame_path)

			targets_slice = target_slices[count]
			for c, target in enumerate(targets_slice):
				new_target_path = os.path.join(target_frame_path, '%05d.npy'%c)
				np.save(new_target_path, target)

			image_frame_path = os.path.join(image_folder_path, '%05d'%(frame_count - count - 1))
			if not os.path.isdir(image_frame_path):
				os.mkdir(image_frame_path)

			print(image_frame_path)

			images_slice = image_slices[count]
			for c, image in enumerate(images_slice):
				new_image_path = os.path.join(image_frame_path, '%05d.png'%c)
				cv2.imwrite(new_image_path, image)

in_folder = '/scratch/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p'
mask_folder = '/scratch/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p'
target_folder = '/scratch/saliency_maps/targets-480p/segments'
image_folder = '/scratch/saliency_maps/images-480p/segments'
slice_size = 10

# train_video_names = ['tractor-sand']
# val_video_names = ['breakdance']
# with open('Training.txt', 'r') as fp:
# 	for name in fp:
# 		train_video_names.append(name.strip())

# with open('Validation.txt', 'r') as fp:
# 	for name in fp:
# 		val_video_names.append(name.strip())

# videos = random.sample(train_video_names, 2)
# for video in videos:
# 	print(video)
save_temporal_slices(in_folder, mask_folder, 'tractor-sand', image_folder, target_folder, slice_size)

# with open('Videos_Training.txt', 'w') as fp:
# 	for name in videos:
# 		fp.write('%s\n'%name)

# videos = random.sample(val_video_names, 2)
# for video in videos:
# 	print(video)
# save_temporal_slices(in_folder, mask_folder, 'breakdance', image_folder, target_folder, slice_size)

# with open('Videos_Validation.txt', 'w') as fp:
# 	for name in videos:
# 		fp.write('%s\n'%name)

# videos = ['lady-running', 'bmx-bumps', 'dog-gooses', 'gold-fish']
# for video in videos:
# 	save_full_sequence(in_folder, mask_folder, video, image_folder, target_folder)