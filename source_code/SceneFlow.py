from __future__ import print_function
import sys
import cv2
import pdb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
from utils.io import mkdir_p
from utils.util_flow import write_flow, save_pfm
from utils.flowlib import point_vec, warp_flow
from utils.flowlib import read_flow, flow_to_image

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='VCN+expansion')
parser.add_argument('--dataset', default='seq',
                    help='KITTI version')
# parser.add_argument('--datapath', default='.//input//Kitti//',
#                     help='dataset path')
parser.add_argument('--datapath', default='E://CVC-14//Day//Visible//NewTest//FramesPos//',
                    help='dataset path')

parser.add_argument('--loadmodel', default='robust.pth',
                    help='model path')
parser.add_argument('--outdir', default='.//output//',
                    help='output dir')
parser.add_argument('--testres', type=float, default=1,
                    help='resolution')
parser.add_argument('--maxdisp', type=int ,default=512,
                    help='maxium disparity. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float ,default=2,
                    help='controls the shape of search grid. Only affect the coarse cost volume size')
args = parser.parse_args()


# dataloader
if args.dataset == '2015':
    from dataloader import kitti15list as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015val':
    from dataloader import kitti15list_val as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015vallidar':
    from dataloader import kitti15list_val_lidar as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015test':
    from dataloader import kitti15list as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'seq':
    from dataloader import seqlist as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sinteltest':
    from dataloader import sintellist as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sintel':
    from dataloader import sintellist_val as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  

max_h = int(maxh // 64 * 64)
max_w = int(maxw // 64 * 64)
if max_h < maxh: max_h += 64
if max_w < maxw: max_w += 64
maxh = max_h
maxw = max_w


mean_L = [[0.33,0.33,0.33]]
mean_R = [[0.33,0.33,0.33]]

# construct model, VCN-expansion
from models.VCN_exp import VCN
model = VCN([1, maxw, maxh], md=[int(4*(args.maxdisp/256)),4,4,4,4], fac=args.fac, 
  exp_unc=('robust' in args.loadmodel))  # expansion uncertainty only in the new model
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('dry run')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


mkdir_p('%s/%s/'% (args.outdir, args.dataset))
def main():
    model.eval()
    ttime_all = []
    for inx in range(len(test_left_img)):
        #print(test_left_img[inx])
        imgL_o = cv2.imread(test_left_img[inx])[:,:,::-1]
        imgR_o = cv2.imread(test_right_img[inx])[:,:,::-1]

        # for gray input images
        if len(imgL_o.shape) == 2:
            imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
            imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))

        # resize
        maxh = imgL_o.shape[0]*args.testres
        maxw = imgL_o.shape[1]*args.testres
        max_h = int(maxh // 64 * 64)
        max_w = int(maxw // 64 * 64)
        if max_h < maxh: max_h += 64
        if max_w < maxw: max_w += 64

        input_size = imgL_o.shape
        imgL = cv2.resize(imgL_o,(max_w, max_h))
        imgR = cv2.resize(imgR_o,(max_w, max_h))

        # flip channel, subtract mean
        imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
        imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
        imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
        imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

        # modify module according to inputs
        from models.VCN_exp import WarpModule, flow_reg
        for i in range(len(model.module.reg_modules)):
            model.module.reg_modules[i] = flow_reg([1,max_w//(2**(6-i)), max_h//(2**(6-i))], 
                            ent=getattr(model.module, 'flow_reg%d'%2**(6-i)).ent,\
                            maxdisp=getattr(model.module, 'flow_reg%d'%2**(6-i)).md,\
                            fac=getattr(model.module, 'flow_reg%d'%2**(6-i)).fac).cuda()
        for i in range(len(model.module.warp_modules)):
            model.module.warp_modules[i] = WarpModule([1,max_w//(2**(6-i)), max_h//(2**(6-i))]).cuda()

        # forward
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        with torch.no_grad():
            imgLR = torch.cat([imgL,imgR],0)
            model.eval()
            torch.cuda.synchronize()
            start_time = time.time()
            rts = model(imgLR)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
            ttime_all.append(ttime)
            flow, occ, logmid, logexp = rts

        # upsampling
        occ = cv2.resize(occ.data.cpu().numpy(),  (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        logexp = cv2.resize(logexp.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        logmid = cv2.resize(logmid.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        flow = torch.squeeze(flow).data.cpu().numpy()
        flow = np.concatenate( [cv2.resize(flow[0],(input_size[1],input_size[0]))[:,:,np.newaxis],
                                cv2.resize(flow[1],(input_size[1],input_size[0]))[:,:,np.newaxis]],-1)
        flow[:,:,0] *= imgL_o.shape[1] / max_w
        flow[:,:,1] *= imgL_o.shape[0] / max_h
        flow = np.concatenate( (flow, np.ones([flow.shape[0],flow.shape[1],1])),-1)
        img = flow_to_image(flow)

        # save predictions
        idxname = test_left_img[inx].split('/')[-1]
        # with open('D:\%s.pfm'% (idxname.split('.')[0]),'w') as f:
        #      print('D:\%s.pfm'% (idxname.split('.')[0]))
        #      save_pfm(f,flow[::-1].astype(np.float32))
        flowvis = point_vec(imgL_o, flow)
        # cv2.imwrite('%s/%s/visflo-%s.jpg'% (args.outdir, args.dataset,idxname),flowvis)
        # cv2.bitwise_not(img,img)
        # cv2.bitwise_not(flowvis, flowvis)
        # cv2.imwrite('E://CVC-14//Test2//%s' % (idxname), flowvis)
        # cv2.imwrite('E://CVC-14//Test3//%s' % (idxname), img)
        print('E://CVC-14//Test2//%s' % (idxname))
        path = 'F://temp2\%s' % (idxname)
        # ori = cv2.imread(path)
        cv2.imshow('original',flowvis)
        cv2.imshow('flow', img)
        cv2.waitKey(1)

        #imwarped = warp_flow(imgR_o, flow[:,:,:2])
        #cv2.imwrite('D:\%s\%s\warp-%s'% (args.outdir, args.dataset,idxname),imwarped[:,:,::-1])

        # with open('%s/%s/occ-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]),'w') as f:
        #     save_pfm(f,occ[::-1].astype(np.float32))
        # with open('%s/%s/exp-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]),'w') as f:
        #     save_pfm(f,logexp[::-1].astype(np.float32))
        # with open('%s/%s/mid-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]),'w') as f:
        #     save_pfm(f,logmid[::-1].astype(np.float32))
        torch.cuda.empty_cache()
    print(np.mean(ttime_all))


class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # type: (str, int, int, int) -> None
        if not name.endswith('.mp4'):  # 保证文件名的后缀是.mp4
            name += '.mp4'
            warnings.warn('video name should ends with ".mp4"')
        self.__name = name          # 文件名
        self.__height = height      # 高
        self.__width = width        # 宽
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 如果是mp4视频，编码需要为mp4v
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()



def main002():
    width = 640
    height = 471
    vw = VideoWriter('E://CVC-14//FILR_Visible.mp4', width, height)
    print(test_left_img)
    for inx in range(len(test_left_img)):
        print(inx)
        #print(test_left_img[inx])
        imgL_o = cv2.imread(test_left_img[inx])[:,:,::-1]
        cv2.imshow('original', imgL_o)
        cv2.waitKey(30)
        vw.write(imgL_o)


if __name__ == '__main__':
    main()