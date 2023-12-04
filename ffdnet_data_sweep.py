import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import FFDNet
from utils import  normalize, remove_dataparallel_wrapper, is_rgb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_ffdnet(**args):

    # Check if input exists and if it is RGB
    try:
        rgb_den = is_rgb(args['input'])
    except:
        raise Exception('Could not open the input image')

    # Open image as a CxHxW torch.Tensor
    if rgb_den:
        in_ch = 3
        model_fn = 'net_rgb.pth'
        imorig = cv2.imread(args['input'])
        #trueim = cv2.imread(args["true_image"])
        # from HxWxC to CxHxW, RGB image
        imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        #trueim = (cv2.cvtColor(trueim, cv2.COLOR_BGR2RGB))
    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        in_ch = 1
        model_fn = 'net_gray.pth'
        imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
        imorig = np.expand_dims(imorig, 0)
        #trueim = cv2.imread(args['true_image'], cv2.IMREAD_GRAYSCALE)
    imorig = np.expand_dims(imorig, 0)
    #trueim = normalize(trueim)

    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = imorig.shape
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
    #print('Loading model ...\n')
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

    # Estimate noise and subtract it to the input image
    im_noise_estim = model(imnoisy, nsigma)
    outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)


    if expanded_h:
        imorig = imorig[:, :, :-1, :]
        outim = outim[:, :, :-1, :]
        imnoisy = imnoisy[:, :, :-1, :]

    if expanded_w:
        imorig = imorig[:, :, :, :-1]
        outim = outim[:, :, :, :-1]
        imnoisy = imnoisy[:, :, :, :-1]

    # Save images
    imnoisy = torch.squeeze(imnoisy,0)
    #imnoisy = torch.squeeze(imnoisy, 0)
    imnoisy = imnoisy.permute([1, 2, 0])
    imnoisy = imnoisy.cpu().numpy()

    outim = torch.squeeze(outim, 0)
    #outim = torch.squeeze(outim, 0)
    outim = outim.permute([1, 2, 0])
    outim = outim.detach().cpu().numpy()

    #trueim = np.expand_dims(trueim,2)
    cv2.imwrite(os.path.join("denoised_images",f"{args['input'][-18:-4]}_d.jpg"), outim*255)
    print(f"{args['input'][-18:-4]}_d.jpg")

challenge_levels = ['/Level_1','/Level_2','/Level_3','/Level_4','/Level_5']
samples_dir = "/home/olives/Desktop/ECE6258_Project/min_samples_2"
data_dir = "/home/olives/Desktop/ECE6258_Project/Datasets"
for file in os.listdir(samples_dir):
    f = open(os.path.join(samples_dir,file), "r")
    images = f.readlines()
    for image in images:
        add_noise = False
        cuda = True
        for i, level in enumerate(challenge_levels):
            input_image = os.path.join(data_dir + '/' + file[:-4] + level, image[:-6]+str(i+1)+'.jpg')
            input_image = input_image.replace('\\','/')
            #input_image = 'D:\\' + file[:-4] + '\\Level_1\\' + image
            no_gpu = False
            noise_sigma = 75
            suffix = ''

            noise_sigma /= 255.
            cuda = not no_gpu and torch.cuda.is_available()

            args = {"add_noise":add_noise,
                    "cuda":cuda,
                    "input":input_image,
                    "no_gpu":no_gpu,
                    "noise_sigma":noise_sigma,
                    "suffix":suffix}

            test_ffdnet(**args)
