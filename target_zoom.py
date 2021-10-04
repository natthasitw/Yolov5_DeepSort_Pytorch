import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sensecam_control import onvif_control
import numpy as np
import pickle as pkl
from numpy import linalg as LA
import math
import time
from Utils import cap_frame, see_center

# RTSP = "rtsp://admin:Hikvisionarv1234@192.168.1.64"
# ip = '192.168.1.64'
# login = 'arvonvif'
# password = 'Arvonvif1234'
with open("PTZ_calibration.pkl", 'rb') as f:
    cam_param = pkl.load(f)
    MTX = cam_param['mtx']
    DIST = cam_param['dist']

with open("PTZ_P_Matrix.pkl", 'rb') as f:
    PTZ_P_Matrix = pkl.load(f)
    rvecs = PTZ_P_Matrix['rvecs']
    tvecs = PTZ_P_Matrix['tvecs']
    Extrinsic = PTZ_P_Matrix['Extrinsic']
    P_Matrix = PTZ_P_Matrix['P_Matrix']
    C_Center = PTZ_P_Matrix['C_Center']

def rad2deg(rad):
    return 180/np.pi*rad


def deg2rad(deg):
    return np.pi/180*deg

class PTZ(object):
    def __init__(self, ip, login, password, P_Matrix, intrinsic):
        self.ip = ip
        self.login = login
        self.password = password
        self.rtsp = "rtsp://admin:Hikvisionarv1234@" + str(self.ip)
        self.cam = onvif_control.CameraControl(ip, login, password)
        self.cam.camera_start()
        self.intrinsic = intrinsic
        self.P_Matrix = P_Matrix
        self.C_Center = np.dot(-np.linalg.inv(self.P_Matrix[:3, :3]), np.array([self.P_Matrix[:, 3]]).T)
        self.P_Matrix_inv = np.linalg.pinv(self.P_Matrix)
        self.PAN0_SHIFT = 0.00556
        self.frame0w = 2560
        self.frame0h = 1440
        # self.frame0 = self.cap_frame()
        # self.cap = cv.VideoCapture(self.rtsp)
        self.home_ptx = self.getptz()


        ###############################
        self.F_LEVEL = np.array([2334.76, 19069.86, 36497.66])  # focal length value list
        self.Z_LEVEL = np.array([0.0, 0.25, 0.5])  # zoom level value list


    def calc_mag(self, inbnbh, targetbnbh):
        if targetbnbh > self.frame0h:
            print("target bounding height can't exceed image height\n "
                  "set mag to 1 (no zoom)")
            return 1
        else:
            return targetbnbh/inbnbh


    def target_zoom(self, imgPt, mag=2):
        pan0, tilt0, zoom0 = self.home_ptx

        u1, v1 = imgPt
        IMG_PT1 = np.array([[self.intrinsic[0, 2], self.intrinsic[1, 2], 1]])
        WORLD_PT1 = np.dot(IMG_PT1, self.P_Matrix_inv.T)
        WORLD_PT1 = np.squeeze(WORLD_PT1 / WORLD_PT1[0, 3])
        ptA = np.squeeze(self.C_Center) - WORLD_PT1[:3]
        normA = LA.norm(ptA[:3])
        ptA = ptA[:3] / normA

        IMG_PT2 = np.array([[u1, v1, 1]])
        WORLD_PT2 = np.dot(IMG_PT2, self.P_Matrix_inv.T)
        WORLD_PT2 = np.squeeze(WORLD_PT2 / WORLD_PT2[0, 3])
        ptB = np.squeeze(self.C_Center) - WORLD_PT2[:3]
        normB = LA.norm(ptB[:3])
        ptB = ptB[:3] / normB

        ## pan
        p = (ptA[2] * ptB[0] - ptA[0] * ptB[2]) / (ptA[0] ** 2 + ptA[2] ** 2)
        pan2 = rad2deg(np.arcsin(p))
        pan_offset = pan2 / 180
        new_pan = pan0 + pan_offset - 0.0004


        ## tilt
        t = ((ptB[2] * ptA[1]) - (ptA[2] * ptB[1])) / (ptA[1] ** 2 + ptA[2] ** 2)
        tilt2 = rad2deg(np.arcsin(t))
        tilt_offset = tilt2 / 55
        new_tilt = tilt0 + tilt_offset - 0.0181

        # zoom interpolate
        fl_z0 = self.F_LEVEL[0]
        fl_z1 = mag * fl_z0
        diff = self.F_LEVEL - fl_z1
        idxA = np.where(diff < 0, diff, -np.inf).argmax()
        idxB = np.where(diff > 0, diff, np.inf).argmin()
        f1 = self.F_LEVEL[idxA]
        f2 = self.F_LEVEL[idxB]
        z1 = self.Z_LEVEL[idxA]
        z2 = self.Z_LEVEL[idxB]
        gap = abs(f1 - f2)
        ratio = (fl_z1 - f1) / gap
        new_zoom = z1 + (ratio * (z2 - z1))

        # adjust for cam_absolute move control
        if new_pan > 1:
            new_pan = -1 + (new_pan - 1)
        if new_pan < -1:
            new_pan = 1 + (new_pan + 1)
        if new_pan < 0:
            new_pan += self.PAN0_SHIFT

        ### Zoom tilt drift ###
        # if new_zoom > 0 and new_zoom <= 0.15:
        #     new_tilt += 0.008
        # if new_zoom > 0.15:
        #     new_tilt += 0.011
        def zoom_tilt_diff(inzoom):
            if inzoom == 0:
                return 0
            else:
                tilt_diff = pow((0.00114405 + inzoom), 0.00203614) - 0.98631007
                return tilt_diff

        ztd = zoom_tilt_diff(new_zoom)
        new_tilt = new_tilt + ztd

        return new_pan, new_tilt, new_zoom


    def absmove(self, pan, tilt, zoom):
        self.cam.absolute_move(pan, tilt, zoom)


    def getptz(self):
        return self.cam.get_ptz()


    def cap_frame(self):
        capture = cv.VideoCapture(self.rtsp)
        ret, frame = capture.read()
        capture.release()
        if ret == True:
            return frame
        else:
            return None


# class PTZslave(PTZ):
#     def __init__(self, ip, login, password, P_Matrix, intrinsic):
#         super().__init__(ip, login, password, P_Matrix, intrinsic)
#
#     def track_person(self, trackduration, ptx, pty, bnbh):
#         mag = self.calc_mag(bnbh, 720)
#         p, t, z = self.target_zoom((ptx, pty), mag)
#         self.absmove(p, t, z)


class FixedCam(object):
    def __init__(self, ip, login, password):
        self.ip = ip
        self.login = login
        self.password = password
        self.rtsp = "rtsp://admin:Hikvisionarv1234@" + str(self.ip)

    def cap_frame(self):
        capture = cv.VideoCapture(self.rtsp)
        ret, frame = capture.read()
        capture.release()
        if ret == True:
            return frame
        else:
            return None


