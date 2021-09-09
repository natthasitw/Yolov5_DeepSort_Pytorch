import cv2 as cv
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

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

HOMO_FIX2PTZ_LMED = np.array([[2.11118231e+00, 2.49756039e-01, -8.80940998e+02],
                              [-2.12270102e-01, 2.05525697e+00, -3.00846856e+02],
                              [4.02324465e-05, 6.59003849e-05, 1.00000000e+00]])

def cap_frame(inRTSP):
    capture = cv.VideoCapture(inRTSP)
    ret, frame = capture.read()
    return ret, frame


def draw_midline(in_frame):
    out_frame = in_frame.copy()
    height, width, _ = out_frame.shape
    half_height = int(height / 2)
    half_width = int(width / 2)
    out_frame = cv.line(out_frame, (half_width, 0),(half_width, height), color=(255,0,0), thickness=2)
    out_frame = cv.line(out_frame, (0, half_height), (width, half_height), color=(255, 0, 0), thickness=2)
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


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def fixed2ptz(x,y,inH):
    pt_1 = np.array([[x,y,1]])
    pt2_tmp = np.dot(inH, pt_1.T)
    x2 = pt2_tmp[0] / pt2_tmp[2]
    y2 = pt2_tmp[1] / pt2_tmp[2]
    return x2[0], y2[0]

inFixMOTtxt = "../data/Hawkeye_Dataset/dual_cam/output_primDetHawk_conf3_iou3/Fix_cam_four_people.txt"
inFixVid = "../data/Hawkeye_Dataset/dual_cam/output_primDetHawk_conf3_iou3/Fix_cam_four_people.mp4"
inFramePTZ = cv.imread("PTZcam_06-09-2021.jpg")
outVidPTZname = 'fix2ptz_tracking.avi'
def write_fix2ptz_tracklet(inFixMOTtxt, inFixVid, inFramePTZ, outVidPTZname, inFixed_Cam_mtx, inFixed_Cam_dist, HOMO_FIX2PTZ_LMED):
    fixtout = []
    with open(inFixMOTtxt) as f:
        for line in f:
            _data = line.rstrip('\n').split(' ')[:-1]
            _data = list(map(lambda x: float(x), _data))
            fixtout.append(_data)
    fixtout = np.array(fixtout)
    fixtout_x = fixtout[:,2] + fixtout[:,4]/2
    fixtout_y = fixtout[:,3] + fixtout[:,5]

    ptz_pts = []
    for x, y in zip(fixtout_x,fixtout_y):
        manual_pts_fixed_undistort = cv.undistortPoints((x, y), inFixed_Cam_mtx, inFixed_Cam_dist, None, inFixed_Cam_mtx)
        x, y = np.squeeze(manual_pts_fixed_undistort)
        newx,newy = fixed2ptz(x, y, HOMO_FIX2PTZ_LMED)
        ptz_pts.append((int(newx),int(newy)))
    ptz_pts = np.array(ptz_pts)
    fixtout = np.hstack((fixtout, ptz_pts))

    vid = cv.VideoCapture(inFixVid)
    vid_fps = vid.get(5)
    vid_totframe = int(vid.get(7))
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    vidwrite = cv.VideoWriter(outVidPTZname, fourcc, vid_fps, (inFramePTZ.shape[1], inFramePTZ.shape[0]))
    for frameiter in range(vid_totframe):
        print(frameiter)
        if frameiter in fixtout[:, 0]:
            tmp_frame = np.copy(inFramePTZ)
            data_row = fixtout[fixtout[:,0] == frameiter]
            for id_row in data_row:
                id_color = compute_color_for_id(id_row[1])
                x = id_row[-2]
                y = id_row[-1]
                if y > inFramePTZ.shape[0] or x > inFramePTZ.shape[1]:
                    continue
                else:
                    tmp_frame = cv.circle(tmp_frame,(int(x),int(y)), 17, id_color, thickness=-1)
            vidwrite.write(tmp_frame)
        else:
            vidwrite.write(inFramePTZ)
    vidwrite.release()

# write_fix2ptz_tracklet(inFixMOTtxt, inFixVid, inFramePTZ, outVidPTZname)