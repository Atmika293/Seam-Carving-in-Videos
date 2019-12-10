from keras.models import Model
from keras import Input
from keras.backend import squeeze, expand_dims
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Activation, Lambda, Embedding, Dense, add, concatenate, LeakyReLU, Dropout #, Reshape, ZeroPadding2D, Cropping2D, Permute
from keras.optimizers import Adadelta

import tensorflow as tf
import numpy as np
import cv2
import os
import random

from targets import generate_target, generate_input_channels

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
allow_growth_session = tf.Session(config=config)
tf.keras.backend.set_session(allow_growth_session)

if tf.test.gpu_device_name():
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
	print("Please install GPU version of TF") 

def downsampling_res_block(x, nb_channels):
    res_path = BatchNormalization()(x)
    res_path = LeakyReLU(alpha=0.3)(res_path)
    res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)
    res_path = MaxPooling2D(pool_size=(2, 2))(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = LeakyReLU(alpha=0.3)(res_path)
    res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)

    shortcut = Conv2D(nb_channels, kernel_size=(1, 1))(x)
    shortcut = MaxPooling2D(pool_size=(2, 2))(shortcut)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path

def upsampling_res_block(x, nb_channels, branch):
	x = UpSampling2D(size=(2, 2))(x)
	upsampled = concatenate([x, branch], axis=3)
	res_path = BatchNormalization()(upsampled)
	res_path = LeakyReLU(alpha=0.3)(res_path)
	res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)
	res_path = BatchNormalization()(res_path)
	res_path = LeakyReLU(alpha=0.3)(res_path)
	res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)

	shortcut = Conv2D(nb_channels, kernel_size=(1, 1))(upsampled)
	shortcut = BatchNormalization()(shortcut)

	res_path = add([shortcut, res_path])
	res_path = Dropout(0.25)(res_path)
	return res_path

def build_model(nb_channels):
	x = Input(shape=(None, None, nb_channels))

	## Padding layer
	## Changing dimensions to multiple of 8, since it is downsampled 3 times by a factor of 2
	# pad_size_ht = 8 - (frame_shape[0] % 8)
	# pad_size_wd = 8 - (frame_shape[1] % 8)
	# top_pad = pad_size_ht // 2
	# bottom_pad = pad_size_ht - top_pad
	# left_pad = pad_size_wd // 2
	# right_pad = pad_size_wd - left_pad

	# padded = ZeroPadding2D(padding=((top_pad, bottom_pad), (left_pad, right_pad)))(x)

	## encoder
	u_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
	u_path = BatchNormalization()(u_path)
	u_path = LeakyReLU(alpha=0.3)(u_path)
	u_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(u_path)

	shortcut = Conv2D(filters=64, kernel_size=(1, 1))(x)
	shortcut = BatchNormalization()(shortcut)
	
	u_path = add([shortcut, u_path])
	
	branch1 = u_path
	u_path = Dropout(0.25)(u_path)

	u_path = downsampling_res_block(u_path, 128)
	branch2 = u_path
	u_path = Dropout(0.25)(u_path)

	## bridge
	u_path = downsampling_res_block(u_path, 256)
	# branch3 = u_path
	u_path = Dropout(0.25)(u_path)

	## bridge
	# u_path = downsampling_res_block(u_path, 512)

	# u_path = Lambda(lambda x: expand_dims(x, axis=0))(u_path)
	# ## Reshape the input to act as a batch for the ConvLSTM2D, which only accepts 5D tensors
	# # shape = u_path.get_shape().as_list()
	# # u_path = Reshape((1, shape[1], shape[2], shape[3]))(u_path)
	# # u_path = Permute((1, 0, 2, 3, 4))(u_path)

	# u_path = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=True, \
	# 	activation='relu', recurrent_activation='relu', dropout=0.1, recurrent_dropout=0.1)(u_path)

	# ## Reshape the output of LSTM to 4D
	# u_path = Lambda(lambda x: squeeze(x, axis=0))(u_path)
	# u_path = Reshape((u_path.shape[0], u_path.shape[1], u_path.shape[2], u_path.shape[3]))

	## decoder
	# u_path = upsampling_res_block(u_path, 256, branch3)
	u_path = upsampling_res_block(u_path, 128, branch2)
	u_path = upsampling_res_block(u_path, 64, branch1)

	output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(u_path)
	
	# output = Cropping2D((top_pad, bottom_pad), (left_pad, right_pad))(u_path)

	model = Model(x, output)

	return model

def generate_batch(in_folder, mask_folder, video, block_ht = 16, block_wd = 16):
	images = []
	targets = []
	video_path = os.path.join(in_folder, video)
	if os.path.isdir(video_path):
		mask_folder_path = os.path.join(mask_folder, video)
					
		frame_count = len(os.listdir(video_path))
		last = None
		for count in range(frame_count):
			frame_path = os.path.join(video_path, '%05d.jpg'%count)
			mask_path = os.path.join(mask_folder_path, '%05d.png'%count)
			# print(frame_path)

			frame = cv2.imread(frame_path, 0) 
			
			if frame.shape[0] > 1080 or frame.shape[1] > 1920:
				return None, None, 0, 0

			images.append(generate_input_channels(frame, last))
			targets.append(generate_target(frame, cv2.imread(mask_path, 0)))

			last = frame

			ht_block_count = frame.shape[0] // block_ht
			wd_block_count = frame.shape[1] // block_wd

	return np.array(images), np.array(targets), ht_block_count, wd_block_count

def train_model(in_folders, mask_folders, block_sizes, channels=3, num_epochs=200, batch_size=10, validation_split=0.1, lr=1, reduce_lr_factor=0.5, loss_delta=0.001, patience=5):
	model = build_model(channels)
	model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['mean_absolute_error']) # default learning rate = 1.0
	# model.summary()
	prefix = 'unet_leakyrelu/binary_crossentropy'

	model.summary()
	model.load_weights(os.path.join(prefix, 'model_epoch_1.h5'))

	video_names = os.listdir(in_folders[0])#list(videos_list[0].keys())

	## split train and validation
	random.shuffle(video_names)
	train_val_split = int((1 - validation_split) * len(video_names))
	train_video_names = video_names[:train_val_split]
	val_video_names = video_names[train_val_split:]

	print(len(train_video_names))
	print(len(val_video_names))

	min_val_loss = 0.0
	best_val_accuracy = 0.0
	prev_val_loss = 0.0
	patience_count = 0

	if not os.path.isfile(os.path.join(prefix, 'Training_statistics.csv')):
		train_stats = open(os.path.join(prefix, 'Training_statistics.csv'), 'w')
		train_stats.write('Epoch,Loss,Metric,%s(Loss),%s(Metric),%s(Loss),%s(Metric),%s(Loss),%s(Metric),%s(Loss),%s(Metric)\n'%('480p','480p','480p(scaled)','480p(scaled)',\
			'1080p(scaled)','1080p(scaled)','1080p','1080p'))
		train_stats.close()

		val_stats = open(os.path.join(prefix, 'Validation_statistics.csv'), 'w')
		val_stats.write('Epoch,Loss,Metric,%s(Loss),%s(Metric),%s(Loss),%s(Metric),%s(Loss),%s(Metric),%s(Loss),%s(Metric)\n'%('480p','480p','480p(scaled)','480p(scaled)',\
			'1080p(scaled)','1080p(scaled)','1080p','1080p'))
		val_stats.close()

		batch_stats = open(os.path.join(prefix, 'Batch_statistics.csv'), 'w')
		batch_stats.write('Epoch,Iteration,Batch,Loss,Metric\n')
		batch_stats.close()

		batch_val_stats = open(os.path.join(prefix, 'Batch_val_statistics.csv'), 'w')
		batch_val_stats.write('Epoch,Iteration,Batch,Loss,Metric\n')
		batch_val_stats.close()
	
	for epoch in range(1, num_epochs):
		iteration = 1

		joint_list = list(zip(in_folders, mask_folders, block_sizes))
		# random.shuffle(joint_list)
		
		total_avg_train_loss = []
		total_avg_train_metric = []
		total_avg_val_loss = []
		total_avg_val_metric = []

		for in_folder, mask_folder, block_size in joint_list:
			print(in_folder, mask_folder, block_size)
			random.shuffle(train_video_names)
			avg_train_loss = 0.0
			avg_train_metric = 0.0
			block_count = 0
			for c, video in enumerate(train_video_names, 1):
				images, targets, ht_block_count, wd_block_count = generate_batch(in_folder, mask_folder, video, block_size, block_size)
				# slice_count = len(images) // slice_size
				# for slice_index in range(slice_count):
				# 	start = 0
				# 	if slice_index < slice_count - 1:
				# 		start = slice_index * slice_size
				# 		X_train = np.array(images[start:(start + slice_size)])
				# 		Y_train = np.array(targets[start:(start + slice_size)])
				# 	else:
				# 		start = slice_index * slice_size
				# 		X_train = np.array(images[start:])
				# 		Y_train = np.array(targets[start:])

				## sliding block window
				for y in range(ht_block_count):
					for x in range(wd_block_count):
						start_y = y * block_size
						start_x = x * block_size

						X_train = images[:, start_y:start_y + block_size, start_x:start_x + block_size, :]
						Y_train = targets[:, start_y:start_y + block_size, start_x:start_x + block_size, :]

						loss, metric = model.train_on_batch(X_train, Y_train)

						block_count += 1

						avg_train_loss += loss
						avg_train_metric += metric

						print('Epoch %d: Iter %d: Video %s: Block (%d, %d): loss = %f, metric = %f'%(epoch + 1, iteration, video, y + 1, x + 1, \
							avg_train_loss / block_count, avg_train_metric / block_count))

				if ht_block_count * wd_block_count > 0:
					print('Epoch %d: Iter %d: Batch %d/%d: loss = %f, metric = %f'%(epoch + 1, iteration, c, len(train_video_names),\
								avg_train_loss / block_count, avg_train_metric / block_count))
					batch_stats = open(os.path.join(prefix, 'Batch_statistics.csv'), 'a')
					batch_stats.write('%d,%d,%d,%.6f,%.6f\n'%(epoch + 1, iteration, c, avg_train_loss / block_count, avg_train_metric / block_count))
					batch_stats.close()
				else:
					print('Epoch %d: Iter %d: Batch %d/%d: skipped'%(epoch + 1, iteration, c, len(train_video_names)))
					
				model.save_weights(os.path.join(prefix, 'batch_model.h5'))

			total_avg_train_loss.append(avg_train_loss / block_count)
			total_avg_train_metric.append(avg_train_metric / block_count)

			random.shuffle(val_video_names)
			avg_val_loss = 0.0
			avg_val_metric = 0.0
			block_count = 0
			for c, video in enumerate(val_video_names, 1):
				images, targets, ht_block_count, wd_block_count = generate_batch(in_folder, mask_folder, video, block_size, block_size)
				# slice_count = len(images) // slice_size
				# for slice_index in range(slice_count):
				# 	start = 0
				# 	if slice_index < slice_count - 1:
				# 		start = slice_index * slice_size
				# 		X_val = np.array(images[start:(start + slice_size)])
				# 		Y_val = np.array(targets[start:(start + slice_size)])
				# 	else:
				# 		start = slice_index * slice_size
				# 		X_val = np.array(images[start:])
				# 		Y_val = np.array(targets[start:])
				## sliding block window
				for y in range(ht_block_count):
					for x in range(wd_block_count):
						start_y = y * block_size
						start_x = x * block_size

						X_val = images[:, start_y:start_y + block_size, start_x:start_x + block_size, :]
						Y_val = targets[:, start_y:start_y + block_size, start_x:start_x + block_size, :]

						loss, metric = model.evaluate(X_val, Y_val, batch_size=X_val.shape[0])

						avg_val_loss += loss
						avg_val_metric += metric

						block_count += 1
						
				if ht_block_count * wd_block_count > 0:
					batch_val_stats = open(os.path.join(prefix, 'Batch_val_statistics.csv'), 'a')
					batch_val_stats.write('%d,%d,%d,%.6f,%.6f\n'%(epoch + 1, iteration, c, avg_val_loss / block_count, avg_val_metric / block_count))
					batch_val_stats.close()

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

			# if prev_val_loss < val_loss + loss_delta:
			# 	patience_count += 1
			# 	if patience_count >= patience:
			# 		lr = lr * reduce_lr_factor
			# 		print('Reducing learning rate to %f'%lr)
			# 		opt = Adadelta(lr=lr)
			# 		model.compile(loss='kld', optimizer=opt, metric='mean_absolute_error')

			# else:
			# 	patience_count = 0

		prev_val_loss = val_loss

def main(): 
	in_folders = ['/scratch/Dataset/DAVIS-2017-trainval-480p/full/images','/scratch/Dataset/DAVIS-2017-trainval-480p/scaled/images']
	mask_folders = ['/scratch/Dataset/DAVIS-2017-trainval-480p/full/masks','/scratch/Dataset/DAVIS-2017-trainval-480p/scaled/masks']
	block_sizes = [16, 16]
	train_model(in_folders, mask_folders, block_sizes)

if __name__=='__main__':
	main()

# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/full/images',
# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/full/masks'
# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/scaled/images'
# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/scaled/masks'
# '/scratch/Dataset/DAVIS-2017-trainval-480p/full/images'
# '/scratch/Dataset/DAVIS-2017-trainval-480p/full/masks'

# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/scaled/images',
# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/full/images'

# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/scaled/masks',
# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/full/masks'