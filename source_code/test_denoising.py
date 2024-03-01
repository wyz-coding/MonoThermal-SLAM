import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import FFDNet
from utils import batch_psnr, normalize, init_logger_ipol, \
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_ffdnet(**args):
	r"""Denoises an input image with FFDNet
	"""
	# Init logger
	logger = init_logger_ipol()

	# Check if input exists and if it is RGB
	try:
		rgb_den = is_rgb(args['input'])
	except:
		raise Exception('Could not open the input image')

	for num in range(1000,2000):
		# Open image as a CxHxW torch.Tensor
		if rgb_den:
			in_ch = 3
			model_fn = 'models/net_thermal_3.pth'
			imorig = cv2.imread(args['input'])

			#输入图像
			path = "./红外图像降噪数据库/红外图像降噪数据8bit库/红外降噪数据库/数据/noise/noise_"+str(num)+".raw"
			path_groud_truth = "./TID/Ground_Truth/gt_" + str(num) + ".png"
			img_gt = cv2.imread(path_groud_truth)
			imorig = np.fromfile(path,dtype=np.uint16)
			imorig = imorig.reshape(192,256)
			# from HxWxC to CxHxW, RGB image
			imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
		else:
			# from HxWxC to  CxHxW grayscale image (C=1)
			in_ch = 1
			model_fn = 'models/net_thermal_1.pth'
			imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
			imorig = np.expand_dims(imorig, 0)
		imorig = np.expand_dims(imorig, 0)

		# Handle odd sizes
		expanded_h = False
		expanded_w = False
		sh_im = imorig.shape
		print(sh_im)
		if sh_im[2]%2 == 1:
			expanded_h = True
			imorig = np.concatenate((imorig, \
					imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

		if sh_im[3]%2 == 1:
			expanded_w = True
			imorig = np.concatenate((imorig, \
					imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

		imorig = normalize(imorig)
		imorig = torch.Tensor(imorig)

		# Absolute path to model file
		model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
					model_fn)

		# Create model
		print('Loading model ...\n')
		net = FFDNet(num_input_channels=in_ch)

		# Load saved weights
		if args['cuda']:
			state_dict = torch.load(model_fn)
			device_ids = [0]
			model = nn.DataParallel(net, device_ids=device_ids).cuda()
		else:
			state_dict = torch.load(model_fn, map_location='cpu')
			# CPU mode: remove the DataParallel wrapper
			state_dict = remove_dataparallel_wrapper(state_dict)
			model = net
		model.load_state_dict(state_dict)

		# Sets the model in evaluation mode (e.g. it removes BN)
		model.eval()

		# Sets data type according to CPU or GPU modes
		if args['cuda']:
			dtype = torch.cuda.FloatTensor
		else:
			dtype = torch.FloatTensor

		# Add noise
		if args['add_noise']:
			noise = torch.FloatTensor(imorig.size()).\
					normal_(mean=0, std=args['noise_sigma'])
			imnoisy = imorig + noise
		else:
			imnoisy = imorig.clone()

			# Test mode
		with torch.no_grad(): # PyTorch v0.4.0
			imorig, imnoisy = Variable(imorig.type(dtype)), \
							Variable(imnoisy.type(dtype))
			nsigma = Variable(
					torch.FloatTensor([args['noise_sigma']]).type(dtype))

		# Measure runtime
		start_t = time.time()

		# Estimate noise and subtract it to the input image
		im_noise_estim = model(imnoisy, nsigma)
		outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)
		stop_t = time.time()

		if expanded_h:
			imorig = imorig[:, :, :-1, :]
			outim = outim[:, :, :-1, :]
			imnoisy = imnoisy[:, :, :-1, :]

		if expanded_w:
			imorig = imorig[:, :, :, :-1]
			outim = outim[:, :, :, :-1]
			imnoisy = imnoisy[:, :, :, :-1]

		# Compute PSNR and log it
		if rgb_den:
			logger.info("### RGB denoising ###")
		else:
			logger.info("### Grayscale denoising ###")
		if args['add_noise']:
			# psnr = batch_psnr(outim, imorig, 1.)
			# print('psnr_original:', psnr)
			print("*********************************")
			# psnr_noisy = batch_psnr(imnoisy, imorig, 1.)
			# print('psnr_noisy',psnr_noisy)
			# path_groud_truth = "./红外图像降噪数据库/红外图像降噪数据8bit库/红外降噪数据库/数据/groundtruth/gt_1500.raw"
			# img_gt = np.fromfile(path_groud_truth,dtype=np.uint16)
			# img_gt = img_gt.reshape(192,256)
			# from HxWxC to CxHxW, RGB image
			# img_gt = (cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
			# psnr_gt = batch_psnr(img_gt, imorig, 1.)
			# print('psnr_gt', psnr_gt)

			#logger.info("\tPSNR_noisy {0:0.2f}dB".format(psnr_noisy))
			#logger.info("\tPSNR_denoised {0:0.2f}dB".format(psnr))
		else:
			logger.info("\tNo noise was added, cannot compute PSNR")
		logger.info("\tRuntime {0:0.4f}s".format(stop_t-start_t))
		print(stop_t-start_t)

		# Compute difference
		diffout   = 2*(outim - imorig) + .5
		diffnoise = 2*(imnoisy-imorig) + .5

		# Save images
		if not args['dont_save_results']:
			noisyimg = variable_to_cv2_image(imnoisy)
			outimg = variable_to_cv2_image(outim)
			ori = variable_to_cv2_image(imorig)

			ori = cv2.flip(ori, 0)  
			ori = cv2.transpose(ori)
			save_path = "./TID/Ground_Truth/gt_"+str(num)+".png"
			# cv2.imwrite(save_path, ori)

			outimg = cv2.flip(outimg, 0) 
			outimg = cv2.transpose(outimg)
			ffd_save_path = "./TID/FFD_25/ffd_" + str(num) + ".png"
			# cv2.imwrite(ffd_save_path, outimg)
			image = np.concatenate((ori,img_gt,outimg), axis=1)
			cv2.imshow("ffdnet", image)
			cv2.waitKey(1)
			num=num=1
			# if args['add_noise']:
			# 	print('have add noise')
				# cv2.imwrite("noisy_diff.png", variable_to_cv2_image(diffnoise))
				# cv2.imwrite("ffdnet_diff.png", variable_to_cv2_image(diffout))

def test_ffdnet_video(**args):
	r"""Denoises an input image with FFDNet
	"""

	class VideoWriter:
		def __init__(self, name, width, height, fps=25):
			# type: (str, int, int, int) -> None
			if not name.endswith('.mp4'):  
				name += '.mp4'
				# warnings.warn('video name should ends with ".mp4"')
			self.__name = name  # 文件名
			self.__height = height  # 高
			self.__width = width  # 宽
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 如果是mp4视频，编码需要为mp4v
			self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

		def write(self, frame):
			if frame.dtype != np.uint8:  # 检查frame的类型
				raise ValueError('frame.dtype should be np.uint8')
			# 检查frame的大小
			row, col, _ = frame.shape
			if row != self.__height or col != self.__width:
				# warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
				return
			self.__writer.write(frame)

	width = 640
	height = 470
	vw = VideoWriter('FL2.mp4', width, height)


	# Init logger
	logger = init_logger_ipol()
	cap = cv2.VideoCapture('./CVC-14/FILR_Day_Test_Frame_Pos.mp4')
	# cap = cv2.VideoCapture(0)
	while cap.isOpened():
		ret, frame = cap.read()
		# test_ffdnet({input: '0025.jpg'})

		# Check if input exists and if it is RGB
		try:
			rgb_den = is_rgb(args['input'])
		except:
			raise Exception('Could not open the input image')

		# Open image as a CxHxW torch.Tensor
		if rgb_den:
			in_ch = 3
			model_fn = 'models/net_thermal_3.pth'
			imorig = cv2.imread(args['input'])
			imorig = frame
			# from HxWxC to CxHxW, RGB image
			imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
		else:
			# from HxWxC to  CxHxW grayscale image (C=1)
			in_ch = 1
			model_fn = 'models/net_thermal_1.pth'
			imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
			imorig = np.expand_dims(imorig, 0)
		imorig = np.expand_dims(imorig, 0)

		# Handle odd sizes
		expanded_h = False
		expanded_w = False
		sh_im = imorig.shape
		# print(sh_im)
		if sh_im[2]%2 == 1:
			expanded_h = True
			imorig = np.concatenate((imorig, \
					imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

		if sh_im[3]%2 == 1:
			expanded_w = True
			imorig = np.concatenate((imorig, \
					imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

		imorig = normalize(imorig)
		imorig = torch.Tensor(imorig)

		# Absolute path to model file
		model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
					model_fn)

		# Create model
		# print('Loading model ...\n')
		net = FFDNet(num_input_channels=in_ch)

		# Load saved weights
		if args['cuda']:
			state_dict = torch.load(model_fn)
			device_ids = [0]
			model = nn.DataParallel(net, device_ids=device_ids).cuda()
		else:
			state_dict = torch.load(model_fn, map_location='cpu')
			# CPU mode: remove the DataParallel wrapper
			state_dict = remove_dataparallel_wrapper(state_dict)
			model = net
		model.load_state_dict(state_dict)

		# Sets the model in evaluation mode (e.g. it removes BN)
		model.eval()

		# Sets data type according to CPU or GPU modes
		if args['cuda']:
			dtype = torch.cuda.FloatTensor
		else:
			dtype = torch.FloatTensor

		# Add noise
		if args['add_noise']:
			noise = torch.FloatTensor(imorig.size()).\
					normal_(mean=0, std=args['noise_sigma'])
			imnoisy = imorig + noise
		else:
			imnoisy = imorig.clone()

			# Test mode
		with torch.no_grad(): # PyTorch v0.4.0
			imorig, imnoisy = Variable(imorig.type(dtype)), \
							Variable(imnoisy.type(dtype))
			nsigma = Variable(
					torch.FloatTensor([args['noise_sigma']]).type(dtype))

		# Measure runtime
		start_t = time.time()

		# Estimate noise and subtract it to the input image
		im_noise_estim = model(imnoisy, nsigma)
		outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)
		stop_t = time.time()

		if expanded_h:
			imorig = imorig[:, :, :-1, :]
			outim = outim[:, :, :-1, :]
			imnoisy = imnoisy[:, :, :-1, :]

		if expanded_w:
			imorig = imorig[:, :, :, :-1]
			outim = outim[:, :, :, :-1]
			imnoisy = imnoisy[:, :, :, :-1]

		# Compute PSNR and log it
		if rgb_den:
			logger.info("### RGB denoising ###")
		else:
			logger.info("### Grayscale denoising ###")
		if args['add_noise']:
			psnr = batch_psnr(outim, imorig, 1.)
			# psnr_noisy = batch_psnr(imnoisy, imorig, 1.)
			print('psnr_original:',psnr)
			# print('psnr_noisy',psnr_noisy)
			#logger.info("\tPSNR_noisy {0:0.2f}dB".format(psnr_noisy))
			#logger.info("\tPSNR_denoised {0:0.2f}dB".format(psnr))
		else:
			logger.info("\tNo noise was added, cannot compute PSNR")
		logger.info("\tRuntime {0:0.4f}s".format(stop_t-start_t))
		print("\tRuntime {0:0.4f}s".format(stop_t-start_t))

		# Compute difference
		diffout   = 2*(outim - imorig) + .5
		diffnoise = 2*(imnoisy-imorig) + .5

		# Save images
		if not args['dont_save_results']:
			noisyimg = variable_to_cv2_image(imnoisy)
			outimg = variable_to_cv2_image(outim)
			# cv2.imwrite("noisy.png", noisyimg)
			# cv2.imwrite("ffdnet_1500.png", outimg)

			if args['add_noise']:
				print('do not anything')
				# cv2.imwrite("noisy_diff.png", variable_to_cv2_image(diffnoise))
				# cv2.imwrite("ffdnet_diff.png", variable_to_cv2_image(diffout))
		cv2.imshow('original',frame)
		cv2.imshow('ffdnet', outimg)
		print(outimg.shape)
		psnr = batch_psnr(outim, imorig, 1.)
		print('psnr',psnr)
		cv2.waitKey(1)
		vw.write(outimg)

	cap.release()
	cv2.destroyAllWindows()


def test_ffdnet_image(**args):
	r"""Denoises an input image with FFDNet
	"""
	# Init logger
	logger = init_logger_ipol()

	# Check if input exists and if it is RGB
	try:
		rgb_den = is_rgb(args['input'])
	except:
		raise Exception('Could not open the input image')

		# Open image as a CxHxW torch.Tensor
	if rgb_den:
		in_ch = 3
		model_fn = 'models/net_thermal_3.pth'
		imorig = cv2.imread('./infrared/right_clahe_0.03.png')
		# imorig = cv2.imread('./infrared/right_0.02.png')
		# from HxWxC to CxHxW, RGB image
		imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		# from HxWxC to  CxHxW grayscale image (C=1)
		in_ch = 1
		model_fn = 'models/net_thermal_1.pth'
		imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
		imorig = np.expand_dims(imorig, 0)
	imorig = np.expand_dims(imorig, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = imorig.shape
	print(sh_im)
	if sh_im[2]%2 == 1:
		expanded_h = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

	if sh_im[3]%2 == 1:
		expanded_w = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

	imorig = normalize(imorig)
	imorig = torch.Tensor(imorig)

	# Absolute path to model file
	model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
				model_fn)

	# Create model
	print('Loading model ...\n')
	net = FFDNet(num_input_channels=in_ch)

	# Load saved weights
	if args['cuda']:
		state_dict = torch.load(model_fn)
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
	else:
		state_dict = torch.load(model_fn, map_location='cpu')
		# CPU mode: remove the DataParallel wrapper
		state_dict = remove_dataparallel_wrapper(state_dict)
		model = net
	model.load_state_dict(state_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model.eval()

	# Sets data type according to CPU or GPU modes
	if args['cuda']:
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor

	# Add noise
	if args['add_noise']:
		noise = torch.FloatTensor(imorig.size()).\
				normal_(mean=0, std=args['noise_sigma'])
		imnoisy = imorig + noise
	else:
		imnoisy = imorig.clone()

		# Test mode
	with torch.no_grad(): # PyTorch v0.4.0
		imorig, imnoisy = Variable(imorig.type(dtype)), \
						Variable(imnoisy.type(dtype))
		nsigma = Variable(
				torch.FloatTensor([args['noise_sigma']]).type(dtype))

	# Measure runtime
	start_t = time.time()

	# Estimate noise and subtract it to the input image
	im_noise_estim = model(imnoisy, nsigma)
	outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)
	stop_t = time.time()

	if expanded_h:
		imorig = imorig[:, :, :-1, :]
		outim = outim[:, :, :-1, :]
		imnoisy = imnoisy[:, :, :-1, :]

	if expanded_w:
		imorig = imorig[:, :, :, :-1]
		outim = outim[:, :, :, :-1]
		imnoisy = imnoisy[:, :, :, :-1]

	# Compute PSNR and log it
	if rgb_den:
		logger.info("### RGB denoising ###")
	else:
		logger.info("### Grayscale denoising ###")
	if args['add_noise']:
		print("*********************************")

	else:
		logger.info("\tNo noise was added, cannot compute PSNR")
	logger.info("\tRuntime {0:0.4f}s".format(stop_t-start_t))
	print(stop_t-start_t)


	# Save images
	if not args['dont_save_results']:
		outimg = variable_to_cv2_image(outim)

		# outimg = cv2.flip(outimg, 0) 
		# outimg = cv2.transpose(outimg)
		cv2.imshow("ffdnet", outimg)
		# cv2.imwrite("./infrared/FFD_right_0.03.png",outimg)
		cv2.waitKey(0)

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="FFDNet_Test")
	parser.add_argument('--add_noise', type=str, default="False")
	parser.add_argument("--input", type=str, default="836.jpg", \
						help='path to input image')
	parser.add_argument("--suffix", type=str, default="", \
						help='suffix to add to output name')
	parser.add_argument("--noise_sigma", type=float, default=15, \
						help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', \
						help="don't save output images")
	parser.add_argument("--no_gpu", action='store_true', \
						help="run model on CPU")
	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	# String to bool
	argspar.add_noise = (argspar.add_noise.lower() == 'true')

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FFDNet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')
	# test_ffdnet_video(**vars(argspar))
	# test_ffdnet(**vars(argspar))
	test_ffdnet_image(**vars(argspar))
	# test_ffdnet({input:'836.jpg'})
