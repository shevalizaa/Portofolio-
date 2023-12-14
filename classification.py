import streamlit as st
import keras
import cv2
from PIL import Image, ImageOps
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

def classification_img(img, weights_file):
    # load model
    device = select_device(device='')
    half=False
    model = DetectMultiBackend(weights = weights_file)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz=(640, 640), s=stride) # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu' # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    #model.warmup(imgsz=(1, 3, *imgsz), half=half) #warmup
    # create the array of the right shape to feed
    # data = np.ndarray(shape=(32,6,6,4), dtype=np.float32)
    # image = img
    # size = (6,6)
    # image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into numpy array
    # image_array = np.asarray(image)
    # normalize the image
    # normalized_image_array = (image_array.astype(np.float32)/127.0) - 1

    # load the image into the array
    # data[0] = normalized_image_array

    # img0 = cv2.imread(img)
    # Padded resize
    im = letterbox(img, imgsz, stride=stride, auto=pt)[0]

    # Convert
    im = im.transpose((2,0,1))[::-1] # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    #im = im.view(32,12,6,6)
    #print(im.size())
    #im2 = impermute(1,0,2,3)
    #run the interface
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)
    #length = pred
    s = ''
    for i, det in enumerate(pred):
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()
            #st.write(int(c))
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            if int(c) == 0:
                if n > 10:
                    st.write('ramai angkot/mikrolet')
            elif int(c) == 1:
                if n > 10:
                    st.write('ramai bus')
            elif int(c) == 2:
                if n > 10:
                    st.write('ramai kendaraan mobil')
            elif int(c) == 3:
                if n > 10:
                    st.write('ramai kendaraan motor')
            elif int(c) == 4:
                if n > 10:
                    st.write('ramai pick-up/box')
            elif int(c) == 5:
                if n > 10:
                    st.write('ramai truk gandar')
    return s