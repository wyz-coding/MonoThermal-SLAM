import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import FFDNet
from utils import batch_psnr, normalize, init_logger_ipol, variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
import tensorflow as tf

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def denoise_frame(frame, args):
    try:
        rgb_den = is_rgb(frame.shape[-3] == 3)
    except:
        raise Exception('Could not process the input image')

    if rgb_den:
        in_ch = 3
        model_fn = 'models/net_thermal_3.pth'
        imorig = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    else:
        in_ch = 1
        model_fn = 'models/net_thermal_1.pth'
        imorig = np.expand_dims(frame, 0)
    imorig = np.expand_dims(imorig, 0)

    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = imorig.shape
    # print(sh_im)
    if sh_im[2] % 2 == 1:
        expanded_h = True
        imorig = np.concatenate((imorig, \
                                 imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

    if sh_im[3] % 2 == 1:
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
        noise = torch.FloatTensor(imorig.size()). \
            normal_(mean=0, std=args['noise_sigma'])
        imnoisy = imorig + noise
    else:
        imnoisy = imorig.clone()

    # Test mode
    with torch.no_grad():  # PyTorch v0.4.0
        imorig, imnoisy = Variable(imorig.type(dtype)), \
                          Variable(imnoisy.type(dtype))
        nsigma = Variable(
            torch.FloatTensor([args['noise_sigma']]).type(dtype))

    # Measure runtime
    start_t = time.time()

    # Estimate noise and subtract it to the input image
    im_noise_estim = model(imnoisy, nsigma)
    outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)
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
        print('psnr_original:', psnr)
    # print('psnr_noisy',psnr_noisy)
    # logger.info("\tPSNR_noisy {0:0.2f}dB".format(psnr_noisy))
    # logger.info("\tPSNR_denoised {0:0.2f}dB".format(psnr))
    else:
        logger.info("\tNo noise was added, cannot compute PSNR")
    logger.info("\tRuntime {0:0.4f}s".format(stop_t - start_t))
    print("\tRuntime {0:0.4f}s".format(stop_t - start_t))

    # Compute difference
    diffout = 2 * (outim - imorig) + .5
    diffnoise = 2 * (imnoisy - imorig) + .5


    noisyimg = variable_to_cv2_image(imnoisy)
    outimg = variable_to_cv2_image(outim)

    return outimg


def callback(data, args):
    bridge = CvBridge()

    try:
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
        return
    outimg = denoise_frame(frame, args)

    try:
        msg_out = bridge.cv2_to_imgmsg(np.array(outimg, dtype=np.uint8), encoding="bgr8")
        pub.publish(msg_out)
    except CvBridgeError as e:
        print(e)


def node():
    rospy.init_node('denoising_node')
    args = {}
    args['cuda'] = False
    args['add_noise'] = False
    args['dont_save_results'] = True
    args['noise_sigma'] = 0

    global pub
    pub = rospy.Publisher('/denoising_image', Image, queue_size=10)

    rospy.Subscriber("/nuc_image", Image, callback, args)
    rospy.spin()


if __name__ == '__main__':
    node()
