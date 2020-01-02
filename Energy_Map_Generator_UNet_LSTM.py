from keras.models import Model
from keras import Input
from keras.backend import squeeze, expand_dims
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Activation, Lambda, Embedding, Dense, add, concatenate, ConvLSTM2D, Dropout #, Reshape, ZeroPadding2D, Cropping2D, Permute
from keras.optimizers import Adadelta

import tensorflow as tf
import numpy as np
import cv2
import os
import random

from targets import generate_inputs_and_targets, generate_inputs_and_targets_for_temporal_slice_for_rnn

## required in case RTX GPU is used
## +++++++
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
allow_growth_session = tf.Session(config=config)
tf.keras.backend.set_session(allow_growth_session)
## +++++++

if tf.test.gpu_device_name():
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
	print("Please install GPU version of TF") 

def downsampling_res_block(x, nb_channels):
    res_path = BatchNormalization()(x)
    res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)
    res_path = MaxPooling2D(pool_size=(2, 2))(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)

    shortcut = Conv2D(nb_channels, kernel_size=(1, 1))(x)
    shortcut = MaxPooling2D(pool_size=(2, 2))(shortcut)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path

def upsampling_res_block(x, nb_channels, branch, dropout):
	x = UpSampling2D(size=(2, 2))(x)
	upsampled = concatenate([x, branch], axis=3)
	res_path = BatchNormalization()(upsampled)
	res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)
	res_path = BatchNormalization()(res_path)
	res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)

	shortcut = Conv2D(nb_channels, kernel_size=(1, 1))(upsampled)
	shortcut = BatchNormalization()(shortcut)

	res_path = add([shortcut, res_path])

	if dropout:
		res_path = Dropout(0.25)(res_path)

	return res_path

def build_model(nb_channels, dropout):
	x = Input(shape=(None, None, nb_channels))

	## encoder
	u_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
	u_path = BatchNormalization()(u_path)
	u_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(u_path)

	shortcut = Conv2D(filters=64, kernel_size=(1, 1))(x)
	shortcut = BatchNormalization()(shortcut)
	
	u_path = add([shortcut, u_path])
	branch1 = u_path
	if dropout:
		u_path = Dropout(0.25)(u_path)

	u_path = downsampling_res_block(u_path, 128)
	branch2 = u_path
	if dropout:
		u_path = Dropout(0.25)(u_path)

	## bridge
	u_path = downsampling_res_block(u_path, 256)

	## Reshape the input to act as a batch for the ConvLSTM2D, which only accepts 5D tensors
	u_path = Lambda(lambda x: expand_dims(x, axis=0))(u_path)

	u_path = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=True, \
		activation='linear', recurrent_activation='linear', dropout=0.1, recurrent_dropout=0.1)(u_path)

	## Reshape the output of LSTM to 4D
	u_path = Lambda(lambda x: squeeze(x, axis=0))(u_path)

	## decoder
	u_path = upsampling_res_block(u_path, 128, branch2, dropout)
	u_path = upsampling_res_block(u_path, 64, branch1, dropout)

	output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(u_path)

	model = Model(x, output)

	return model

## generates a batch for training the network
## Each video is passed as a batch of temporal slices. The batch size is the number of frames in the video.
## Each slice has nth frame to (n + slice_size - 1)th frame in reverse order. Same for the target slices.
## The input temporal slices in each batch consists of a list of 3-channel inputs, each corresponding to one frame in the video.
## The target temporal slices in each batch is a list of saliency maps, each corresponding to one frame in the video.
def create_temporal_slices(in_folder, mask_folder, video, target_folder, slice_size):
	frames = []
	masks = []

	# inputs = []
	input_slices = []
	targets = []
	target_slices = []

	video_path = os.path.join(in_folder, video)
	if os.path.isdir(video_path):
		mask_folder_path = os.path.join(mask_folder, video)
		target_folder_path = os.path.join(target_folder, video)
					
		frame_count = len(os.listdir(video_path))
		for count in range(frame_count):
			frame_path = os.path.join(video_path, '%05d.jpg'%count)
			mask_path = os.path.join(mask_folder_path, '%05d.png'%count)

			frame = cv2.imread(frame_path, 0)
			mask = cv2.imread(mask_path, 0)

			frames.append(frame)
			masks.append(mask)

		if os.path.isdir(target_folder_path):
			input_slices, _ = generate_inputs_and_targets_for_temporal_slice_for_rnn(frames, masks, slice_size)

			for count in range(frame_count - 1, -1, -1):
				target_frame_path = os.path.join(target_folder_path, '%05d'%count)

				targets = []
				slice_size = len(os.listdir(target_frame_path))
				for c in range(slice_size):
					target_path = os.path.join(target_frame_path, '%05d.npy'%c)					
					target = np.load(target_path)
					targets.append(np.expand_dims(target, axis=2))

				target_slices.append(np.array(targets))

		else:
			input_slices, target_slices = generate_inputs_and_targets_for_temporal_slice_for_rnn(frames, masks, slice_size, generate_targets=True)

			if not os.path.isdir(target_folder_path):
				os.mkdir(target_folder_path)
			
			for count in range(frame_count):
				target_frame_path = os.path.join(target_folder_path, '%05d'%(frame_count - count - 1))
				if not os.path.isdir(target_frame_path):
					os.mkdir(target_frame_path)

				targets_slice = target_slices[count]
				slice_size = targets_slice.shape[0]
				for c in range(slice_size):
					new_target_path = os.path.join(target_frame_path, '%05d.npy'%c)
					np.save(new_target_path, np.squeeze(targets_slice[c, :, :, :], axis=2))

	return input_slices, target_slices

## channels: number of channels in each input in the temporal slice
## slice_size: size of temporal slice to be trained on.
## validation_split: percentage of dataset to be alloted to validation set. 
def train_model(in_folders, mask_folders, target_folders, channels=3, slice_size=10, dropout=False, loss_fn='mean_absolute_error', metric_fn=['binary_crossentropy'], \
				num_epochs=100, split_data=True, validation_split=0.1, folder_prefix='/scratch/56x32/unet_lstm/mae_wo_dropout'):
	model = build_model(channels, dropout)
	model.compile(loss=loss_fn, optimizer='Adadelta', metrics=metric_fn) # default learning rate = 1.0
	model.summary()

	prefix = folder_prefix
	train_video_names = []
	val_video_names = []
	if split_data:
		video_names = os.listdir(in_folders[0])
		## split train and validation
		random.shuffle(video_names)
		train_val_split = int((1 - validation_split) * len(video_names))
		train_video_names = video_names[:train_val_split]
		val_video_names = video_names[train_val_split:]

		with open(os.path.join(prefix, 'Training.txt'), 'w') as fp:
			for name in train_video_names:
				fp.write('%s\n'%name)

		with open(os.path.join(prefix, 'Validation.txt'), 'w') as fp:
			for name in val_video_names:
				fp.write('%s\n'%name)

	else:
		with open(os.path.join(prefix, 'Training.txt'), 'r') as fp:
			for name in fp:
				train_video_names.append(name.strip())

		with open(os.path.join(prefix, 'Validation.txt'), 'r') as fp:
			for name in fp:
				val_video_names.append(name.strip())

	print(len(train_video_names))
	print(len(val_video_names))
	print(prefix)

	min_val_loss = 0.0
	best_val_accuracy = 0.0
	patience_count = 0

	if not os.path.isfile(os.path.join(prefix, 'Training_statistics.csv')):
		train_stats = open(os.path.join(prefix, 'Training_statistics.csv'), 'w')
		train_stats.write('Epoch,Loss,Metric,%s(Loss),%s(Metric)\n'%('480p(scaled)','480p(scaled)'))
		train_stats.close()

		val_stats = open(os.path.join(prefix, 'Validation_statistics.csv'), 'w')
		val_stats.write('Epoch,Loss,Metric,%s(Loss),%s(Metric)\n'%('480p(scaled)','480p(scaled)'))
		val_stats.close()
	
	for epoch in range(num_epochs):
		iteration = 1

		joint_list = list(zip(in_folders, mask_folders, target_folders))
		# random.shuffle(joint_list)
		
		total_avg_train_loss = []
		total_avg_train_metric = []
		total_avg_val_loss = []
		total_avg_val_metric = []

		for in_folder, mask_folder, target_folder in joint_list:
			print(in_folder, mask_folder, target_folder)
			random.shuffle(train_video_names)
			avg_train_loss = 0.0
			avg_train_metric = 0.0
			block_count = 0
			for c, video in enumerate(train_video_names, 1):
				## sliding temporal window
				video_slices, target_slices = create_temporal_slices(in_folder, mask_folder, video, target_folder, slice_size=slice_size)
				for video_slice, targets_slice in zip(video_slices, target_slices):
					loss, metric = model.train_on_batch(video_slice, targets_slice)
					avg_train_loss += loss
					avg_train_metric += metric
					block_count += 1

				model.save_weights(os.path.join(prefix, 'batch_model.h5'))
				print('Epoch %d: Iter %d: Batch %d/%d: loss = %f, metric = %f'%(epoch + 1, iteration, c, len(train_video_names),\
					avg_train_loss / block_count, avg_train_metric / block_count))

			total_avg_train_loss.append(avg_train_loss / block_count)
			total_avg_train_metric.append(avg_train_metric / block_count)

			random.shuffle(val_video_names)
			avg_val_loss = 0.0
			avg_val_metric = 0.0
			block_count = 0
			for c, video in enumerate(val_video_names, 1):
				## sliding temporal window
				video_slices, target_slices = create_temporal_slices(in_folder, mask_folder, video, target_folder, slice_size=slice_size)
				for video_slice, targets_slice in zip(video_slices, target_slices):
					loss, metric = model.train_on_batch(video_slice, targets_slice)
					avg_val_loss += loss
					avg_val_metric += metric
					block_count += 1

			print('Epoch %d: Iter %d: loss = %f, metric = %f'%(epoch + 1, iteration, avg_val_loss / block_count, avg_val_metric / block_count))

			total_avg_val_loss.append(avg_val_loss / block_count)
			total_avg_val_metric.append(avg_val_metric / block_count)

			iteration += 1

		model.save_weights(os.path.join(prefix, 'model_epoch_%d.h5'%(epoch + 1)))
		
		train_stats = open(os.path.join(prefix, 'Training_statistics.csv'), 'a')
		train_stats.write('%.d,%.6f,%.6f'%(epoch + 1, sum(total_avg_train_loss) / len(total_avg_train_loss), sum(total_avg_train_metric) / len(total_avg_train_metric)))
		for train_loss, train_metric in zip(total_avg_train_loss, total_avg_train_metric):
			train_stats.write(',%.6f,%.6f'%(train_loss, train_metric))
		train_stats.write('\n')
		train_stats.close()

		val_stats = open(os.path.join(prefix, 'Validation_statistics.csv'), 'a')
		val_stats.write('%.d,%.6f,%.6f'%(epoch + 1, sum(total_avg_val_loss) / len(total_avg_val_loss), sum(total_avg_val_metric) / len(total_avg_val_metric)))
		for val_loss, val_metric in zip(total_avg_val_loss, total_avg_val_metric):
			val_stats.write(',%.6f,%.6f'%(val_loss, val_metric))
		val_stats.write('\n')
		val_stats.close()

		val_loss = sum(total_avg_val_loss) / len(total_avg_val_loss)
		val_acc = sum(total_avg_val_metric) / len(total_avg_val_metric)

		if epoch == 0:
			min_val_loss = val_loss
			best_val_accuracy = val_acc
		else:
			if min_val_loss >= val_loss:
				min_val_loss = val_loss
				model.save_weights(os.path.join(prefix, 'model_least_loss.h5'))

			if best_val_accuracy >= val_acc:
				best_val_accuracy = val_acc
				model.save_weights(os.path.join(prefix, 'model_best_acc.h5'))

## in_folders: list of folders which contain the video folders
## mask_folders: list of folders which contain the corresponding segmentation masks for the video folders
## out_folders: list of folders where the target saliency maps generated for training will be saved. 
## slice_size: size of temporal slice to be trained on.
## dropout: if True, Dropout layers are included in the model
## loss_fn: loss function to be used
## metric_fn: list of metric functions to be used
## num_epochs: number of epochs to run training for
## split_data: if True, splits dataset into training and validation sets. Else, it looks for 'Training.txt' and 'Validation.txt' 
## in the 'folder_prefix' folder. These files must contain a list of training and validation videos respectively.
## The list of training and validation videos are saved to 'Training.txt' and 'Validation.txt', if they are not already present.
## folder_prefix: the folder where the model weights are saved after every epoch.
def train(): 
	in_folders = ['/scratch/Dataset/DAVIS-2017-trainval-480p/56x32/images']
	mask_folders = ['/scratch/Dataset/DAVIS-2017-trainval-480p/56x32/masks']
	out_folders = ['/scratch/Dataset/DAVIS-2017-trainval-480p/56x32/temporal_slices/30']
	train_model(in_folders, mask_folders, out_folders, slice_size=30, dropout=True, loss_fn='mean_absolute_error', \
		metric_fn=['binary_crossentropy'], num_epochs=50, split_data=False, folder_prefix='/scratch/56x32/unet_lstm/mae')

if __name__=='__main__':
	train()