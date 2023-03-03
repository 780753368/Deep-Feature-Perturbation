import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset
# from util import *
import scipy.misc
import torchfile
# from torch.utils.serialization import load_lua
import time
import util

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath', default='images/content', help='path to train')
parser.add_argument('--stylePath', default='images/style', help='path to train')
parser.add_argument('--model', default='wct.pt', help='Path to model')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=512,
                    help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='samples/', help='folder to output images')
parser.add_argument('--alpha', type=float, default=0.8, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")
args = parser.parse_args()
try:
    os.makedirs(args.outf)
except OSError:
    pass

torch.backends.cudnn.benchmark = True

# Data loading code
content_dataset = Dataset(args.contentPath, args.fineSize)
content_loader = torch.utils.data.DataLoader(dataset=content_dataset, batch_size=1, shuffle=False)

style_dataset = Dataset(args.stylePath, args.fineSize)
style_loader = torch.utils.data.DataLoader(dataset=style_dataset, batch_size=1, shuffle=False)

# wct = util.WCT(args.model)
wct = torch.load(args.model)
# torch.save(wct, 'wct.pt')

def styleTransfer(contentImg, styleImg, content_name, style_name, csF):
    sF5 = wct.e5(styleImg)
    cF5 = wct.e5(contentImg)
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    csF5 = wct.transform(cF5, sF5, csF, args.alpha)
    Im5 = wct.d5(csF5)

    sF4 = wct.e4(styleImg)
    cF4 = wct.e4(Im5)
    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    csF4 = wct.transform(cF4, sF4, csF, args.alpha)
    Im4 = wct.d4(csF4)

    sF3 = wct.e3(styleImg)
    cF3 = wct.e3(Im4)
    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = wct.transform(cF3, sF3, csF, args.alpha)
    Im3 = wct.d3(csF3)

    sF2 = wct.e2(styleImg)
    cF2 = wct.e2(Im3)
    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = wct.transform(cF2, sF2, csF, args.alpha)
    Im2 = wct.d2(csF2)

    sF1 = wct.e1(styleImg)
    cF1 = wct.e1(Im2)
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = wct.transform(cF1, sF1, csF, args.alpha)
    Im1 = wct.d1(csF1)
    # save_image has this wired design to pad images with 4 pixels at default.
    name = content_name[:-4] + '_to_' + style_name[:-4] + '_' + str(args.alpha) + '.jpg'
    vutils.save_image(Im1.data.cpu().float(), os.path.join(args.outf, name))
    return


avgTime = 0
cImg = torch.Tensor()
sImg = torch.Tensor()
csF = torch.Tensor()
csF = Variable(csF)
if (args.cuda):
    cImg = cImg.cuda(args.gpu)
    sImg = sImg.cuda(args.gpu)
    csF = csF.cuda(args.gpu)
    wct.cuda(args.gpu)

# content = iter(content_loader)
# content_img, content_name = next(content)
# style = iter(style_loader)
# style_img, style_name = next(style)
# print('Transferring ' + ''.join(content_name))
# if (args.cuda):
#     content_img = content_img.cuda(args.gpu)
#     style_img = style_img.cuda(args.gpu)
# cImg = Variable(content_img, volatile=True)
# sImg = Variable(style_img, volatile=True)
# start_time = time.time()
# # WCT Style Transfer
# styleTransfer(cImg, sImg, ''.join(content_name), ''.join(style_name), csF)
# end_time = time.time()
# print('Elapsed time is: %f' % (end_time - start_time))
# avgTime += (end_time - start_time)

for i, (content_img, content_name) in enumerate(content_loader):
    for j, (style_img, style_name) in enumerate(style_loader):
        print(''.join(content_name) + ' transferring to ' + ''.join(style_name))
        if (args.cuda):
            content_img = content_img.cuda(args.gpu)
            style_img = style_img.cuda(args.gpu)
        cImg = Variable(content_img, volatile=True)
        sImg = Variable(style_img, volatile=True)
        start_time = time.time()
        # WCT Style Transfer
        styleTransfer(cImg, sImg, ''.join(content_name), ''.join(style_name), csF)
        end_time = time.time()
        print('Elapsed time is: %f' % (end_time - start_time))
        avgTime += (end_time - start_time)
