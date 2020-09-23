from random import randint
from PIL import Image
import numpy as np
import cv2
import pprint
import pycuda.driver as cuda
import common
import pycuda.autoinit
import tensorrt as trt
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = common.GiB(1)
        builder.max_batch_size = 1
        parser.register_input('input_2:0', (3, 299, 299))
        parser.parse(model_file, network)

        return builder.build_cuda_engine(network)
