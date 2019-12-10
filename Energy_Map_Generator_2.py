from keras.models import Model
from keras import Input
from keras.backend import squeeze, expand_dims
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Activation, Lambda, Embedding, Dense, add, concatenate, Dropout #, Reshape, ZeroPadding2D, Cropping2D, Permute
from keras.optimizers import Adadelta

import tensorflow as tf
import numpy as np
import cv2
import os
import random

from targets import generate_target, propagate_importance, generate_input_channels

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
    res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)
    res_path = MaxPooling2D(pool_size=(2, 2))(res_path)
    res_path = BatchNormalization()(res_path)
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
	res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)
	res_path = BatchNormalization()(res_path)
	res_path = Conv2D(filters=nb_channels, kernel_size=(3, 3), padding='same')(res_path)

	shortcut = Conv2D(nb_channels, kernel_size=(1, 1))(upsampled)
	shortcut = BatchNormalization()(shortcut)

	res_path = add([shortcut, res_path])
	res_path = Dropout(0.1)(res_path)
	
	return res_path

def build_model(nb_channels):
	x = Input(shape=(None, None, nb_channels))

	## encoder
	u_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
	u_path = BatchNormalization()(u_path)
	u_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(u_path)

	shortcut = Conv2D(filters=64, kernel_size=(1, 1))(x)
	shortcut = BatchNormalization()(shortcut)
	
	u_path = add([shortcut, u_path])
	branch1 = u_path
	u_path = Dropout(0.1)(u_path)

	u_path = downsampling_res_block(u_path, 128)
	branch2 = u_path
	u_path = Dropout(0.1)(u_path)

	## bridge
	u_path = downsampling_res_block(u_path, 256)
	u_path = Dropout(0.1)(u_path)

	## decoder
	u_path = upsampling_res_block(u_path, 128, branch2)
	u_path = upsampling_res_block(u_path, 64, branch1)

	output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(u_path)

	model = Model(x, output)

	return model

def generate_batch(in_folder, mask_folder, video, out_folder, block_ht = 16, block_wd = 16):
	images = []
	targets = []
	frames = []

	video_path = os.path.join(in_folder, video)
	if os.path.isdir(video_path):
		mask_folder_path = os.path.join(mask_folder, video)
		target_folder_path = os.path.join(out_folder, video)

		if not os.path.isdir(target_folder_path):
			os.mkdir(target_folder_path)
					
		frame_count = len(os.listdir(video_path))
		last = None
		for count in range(frame_count):
			frame_path = os.path.join(video_path, '%05d.jpg'%count)
			mask_path = os.path.join(mask_folder_path, '%05d.png'%count)
			# print(frame_path)

			frame = cv2.imread(frame_path, 0) 
			
			if frame.shape[0] > 1080 or frame.shape[1] > 1920:
				return None, None, 0, 0

			frames.append(frame)

			images.append(generate_input_channels(frame, last))

			target_path = os.path.join(target_folder_path, '%05d.npy'%count)
			if os.path.isfile(target_path):
				targets.append(np.expand_dims(np.load(target_path), axis=2))
			else:
				targets.append(generate_target(frame, cv2.imread(mask_path, 0)))

				if len(targets) == frame_count:
					targets = propagate_importance(frames, targets)

					for count, target in enumerate(targets):
						new_target_path = os.path.join(target_folder_path, '%05d.npy'%count)
						np.save(new_target_path, np.squeeze(target, axis=2))

			last = frame

			ht_block_count = frame.shape[0] // block_ht
			wd_block_count = frame.shape[1] // block_wd

	return np.array(images), np.array(targets), ht_block_count, wd_block_count

def train_model(in_folders, mask_folders, block_sizes, out_folders, channels=3, num_epochs=200, batch_size=10, validation_split=0.1, lr=1, reduce_lr_factor=0.5, loss_delta=0.001, patience=5):
	model = build_model(channels)
	model.compile(loss='mean_absolute_error', optimizer='Adadelta', metrics=['binary_crossentropy']) # default learning rate = 1.0
	prefix = 'unet_prop/mae/'

	model.summary()
	model.load_weights(os.path.join('unet_prop/mae', 'model_epoch_1.h5'))

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

		joint_list = list(zip(in_folders, mask_folders, out_folders, block_sizes))
		
		total_avg_train_loss = []
		total_avg_train_metric = []
		total_avg_val_loss = []
		total_avg_val_metric = []

		for in_folder, mask_folder, out_folder, block_size in joint_list:
			print(in_folder, mask_folder, out_folder, block_size)
			random.shuffle(train_video_names)
			avg_train_loss = 0.0
			avg_train_metric = 0.0
			block_count = 0
			for c, video in enumerate(train_video_names, 1):
				images, targets, ht_block_count, wd_block_count = generate_batch(in_folder, mask_folder, video, out_folder, block_size, block_size)

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

			# random.shuffle(val_video_names)

			avg_val_loss = 0.0
			avg_val_metric = 0.0
			block_count = 0
			for c, video in enumerate(val_video_names, 1):
				images, targets, ht_block_count, wd_block_count = generate_batch(in_folder, mask_folder, video, out_folder, block_size, block_size)

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

def generate_saliency_map(in_folder, out_folder, block_size=16):
	model = build_model(3)
	model.compile(loss='mean_absolute_error', optimizer='Adadelta', metrics=['binary_crossentropy']) # default learning rate = 1.0

	model.summary()
	model.load_weights(os.path.join('unet/mae', 'model_epoch_19.h5'))

	frame_count = len(os.listdir(in_folder))
	last = None

	for count in range(frame_count):
		frame_path = os.path.join(in_folder, '%05d.jpg'%count)
		frame = cv2.imread(frame_path, 0) 

		ht_block_count = frame.shape[0] // block_size
		wd_block_count = frame.shape[1] // block_size

		image = np.array([generate_input_channels(frame, last)])
		saliency_map = np.zeros([frame.shape[0], frame.shape[1], 1], dtype=np.float64)

		for y in range(ht_block_count):
			for x in range(wd_block_count):
				start_y = y * block_size
				start_x = x * block_size
				X_val = image[:, start_y:start_y + block_size, start_x:start_x + block_size, :]

				saliency_map[start_y:start_y + block_size, start_x:start_x + block_size, :] = model.predict(X_val, batch_size=1)

		np.save(os.path.join(out_folder, '%05d.npy'%count), np.squeeze(saliency_map, axis=2))

		last = frame


def main(): 
	in_folders = ['/scratch/Dataset/DAVIS-2017-trainval-480p/full/images']
	mask_folders = ['/scratch/Dataset/DAVIS-2017-trainval-480p/full/masks']
	out_folders = ['/scratch/Dataset/DAVIS-2017-trainval-480p/full/targets']
	block_sizes = [16] #, 16
	train_model(in_folders, mask_folders, block_sizes, out_folders)

if __name__=='__main__':
	main()
	# generate_saliency_map('/scratch/Dataset/DAVIS-2017-trainval-480p/full/images/tractor-sand', '/scratch/network_output/unet')

	# files = os.listdir('/scratch/network_output/unet')
	# for file in files:
	# 	if file.endswith('.npy'):
	# 		path = os.path.join('/scratch/network_output/unet', file)
	# 		saliency_map = np.load(path)
	# 		# saliency_map[saliency_map > 1] = 1.0
	# 		# saliency_map[saliency_map < 0] = 0.0
	# 		cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
	# 		cv2.imwrite(os.path.join('/scratch/network_output/unet/images', file.split('.')[0] + '.png'), np.uint8(saliency_map * 255))

# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/full/images',
# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/full/masks'
# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/scaled/images'
# '/scratch/Dataset/DAVIS-2017-trainval-Full-Resolution/scaled/masks'
# '/scratch/Dataset/DAVIS-2017-trainval-480p/full/images'
# '/scratch/Dataset/DAVIS-2017-trainval-480p/full/masks'