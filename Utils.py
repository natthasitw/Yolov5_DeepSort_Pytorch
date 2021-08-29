import cv2
from matplotlib import pyplot as plt
import json
from sensecam_control import onvif_control
from sensecam_control import onvif_config
import pandas as pd
import time
from datetime import datetime
import os
from tqdm import tqdm
import numpy as np



def cap_frame(inRTSP):
    capture = cv2.VideoCapture(inRTSP)
    ret, frame = capture.read()
    return ret, frame


def draw_midline(in_frame):
    out_frame = in_frame.copy()
    height, width, _ = out_frame.shape
    half_height = int(height / 2)
    half_width = int(width / 2)
    out_frame = cv2.line(out_frame, (half_width, 0),(half_width, height), color=(255,0,0), thickness=2)
    out_frame = cv2.line(out_frame, (0, half_height), (width, half_height), color=(255, 0, 0), thickness=2)
    return out_frame


def see_center(inRTSP):
    _, frame = cap_frame(inRTSP)
    frame = frame[:,:,[2,1,0]]
    out_frame = draw_midline(frame)
    return out_frame


def rad2deg(rad):
    return 180/np.pi*rad


def deg2rad(deg):
    return np.pi/180*deg