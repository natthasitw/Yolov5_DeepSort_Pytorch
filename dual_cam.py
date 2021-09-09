import sys
sys.path.insert(0, './yolov5')
from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow,xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5.utils.plots import plot_one_box
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from zone import ROI, ROIS
from sensecam_control import onvif_control
from matplotlib import pyplot as plt
from Utils import cap_frame, see_center
import pickle as pkl
from collections import deque
import yaml
from Utils import *
from target_zoom import *


with open('config_dualcam.yaml') as f:
    config_data = yaml.load(f, Loader=yaml.FullLoader)
    cam_list = ['Fixed_Cam', 'PTZ_Cam']
    cam_attributes = ['IP', 'login', 'password', 'mtx', 'dist']
    Fixed_Cam_IP = config_data['Fixed_Cam']['IP']
    Fixed_Cam_login = config_data['Fixed_Cam']['login']
    Fixed_Cam_password = config_data['Fixed_Cam']['password']
    Fixed_Cam_mtx = np.array(config_data['Fixed_Cam']['mtx'])
    Fixed_Cam_dist = np.array(config_data['Fixed_Cam']['dist'])
    PTZ_Cam_IP = config_data['PTZ_Cam']['IP']
    PTZ_Cam_login = config_data['PTZ_Cam']['login']
    PTZ_Cam_password = config_data['PTZ_Cam']['password']
    PTZ_Cam_mtx = np.array(config_data['PTZ_Cam']['mtx'])
    PTZ_Cam_dist = np.array(config_data['PTZ_Cam']['dist'])
    PTZ_Cam_PMatrix = np.array(config_data['PTZ_Cam']['P_Matrix'])
    HOMO_FIX2PTZ_LMED = config_data['H_FIX2PTZ']


fix_cam = FixedCam(Fixed_Cam_IP, Fixed_Cam_login, Fixed_Cam_password)
ptz_cam = PTZ(PTZ_Cam_IP, PTZ_Cam_login, PTZ_Cam_password, PTZ_Cam_PMatrix, PTZ_Cam_mtx)

# RTSP_fix = "rtsp://admin:Hikvisionarv1234@192.168.1.65"
# RTSP_ptz = "rtsp://admin:Hikvisionarv1234@192.168.1.64"
# ip_fix = '192.168.1.65'
# ip_ptz = '192.168.1.64'
# login = 'arvonvif'
# password = 'Arvonvif1234'
#
# with open("PTZ_calibration.pkl", 'rb') as f:
#     cam_param = pkl.load(f)
#     MTX = cam_param['mtx']
#     DIST_ptz = cam_param['dist']
# with open("PTZ_P_Matrix.pkl", 'rb') as f:
#     PTZ_P_Matrix = pkl.load(f)
#     P_Matrix = PTZ_P_Matrix['P_Matrix']

# fix_cam = FixedCam(Fixed_Cam_IP, Fixed_Cam_login, Fixed_Cam_password)
# ptz_cam = PTZ(PTZ_Cam_IP, PTZ_Cam_login, PTZ_Cam_password, PTZ_Cam_PMatrix, PTZ_Cam_mtx)

# palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
#
# def compute_color_for_id(label):
#     """
#     Simple function that adds fixed color depending on the id
#     """
#     palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
#
#     color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
#     return tuple(color)
#
# def xyxy_to_xywh(*xyxy):
#     """" Calculates the relative bounding box from absolute pixel values. """
#     bbox_left = min([xyxy[0].item(), xyxy[2].item()])
#     bbox_top = min([xyxy[1].item(), xyxy[3].item()])
#     bbox_w = abs(xyxy[0].item() - xyxy[2].item())
#     bbox_h = abs(xyxy[1].item() - xyxy[3].item())
#     x_c = (bbox_left + bbox_w / 2)
#     y_c = (bbox_top + bbox_h / 2)
#     w = bbox_w
#     h = bbox_h
#     return x_c, y_c, w, h
#
# def xyxy_to_tlwh(bbox_xyxy):
#     tlwh_bboxs = []
#     for i, box in enumerate(bbox_xyxy):
#         x1, y1, x2, y2 = [int(i) for i in box]
#         top = x1
#         left = y1
#         w = int(x2 - x1)
#         h = int(y2 - y1)
#         tlwh_obj = [top, left, w, h]
#         tlwh_bboxs.append(tlwh_obj)
#     return tlwh_bboxs
#
#
# def compute_color_for_labels(label):
#     """
#     Simple function that adds fixed color depending on the class
#     """
#     color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
#     return tuple(color)
#
#
# def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
#     for i, box in enumerate(bbox):
#         x1, y1, x2, y2 = [int(i) for i in box]
#         x1 += offset[0]
#         x2 += offset[0]
#         y1 += offset[1]
#         y2 += offset[1]
#         # box text and bar
#         id = int(identities[i]) if identities is not None else 0
#         color = compute_color_for_labels(id)
#         label = '{}{:d}'.format("", id)
#         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#         cv2.rectangle(
#             img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
#         cv2.putText(img, label, (x1, y1 +
#                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
#     return img

class TracksROIControl(ROIS):
    def __init__(self, rois_polygon):
        '''
        track_buffer: ID, bnbx, bnby, bnbw, bnbh, ROIid, tracked_flag
        track_buffer_v2: ID, bnbx, bnby, bnbw, bnbh, ROIid, missed_counter, tracked_flag
        '''
        self.track_id = deque(maxlen=int(20))
        self.track_buffer = deque(maxlen=int(20))
        self.tztracking_state = False
        self.current_tzt_id = 0  # only adjustable in init_tzt_onID()
        self.tzt_duration = 25  # target zoom tracking duration in frames
        self.tzt_counter = 0
        self.missed_track_thresh = 10  # the max # of frames of ID that can be missed during tzt.
        self.rois = ROIS(rois_polygon)


    def _update_tracks(self, outputs_v):
        '''
        Must run in for loop for each frame
        Parameters
        ----------
        outputs_v: from the deep_sort_2.py update() method

        Returns: Dont return anything. Just update the self.track_buffer
        -------

        '''

        # Increment missed_counter by 1 for all items in track_buffer
        for i, track_buf in enumerate(self.track_buffer):
            if track_buf[-1] == 0:
                self.track_buffer[i][6] = track_buf[6] + 1

        for output in outputs_v:
            x1, y1, x2, y2 = output[0:4]
            bnbw = x2-x1
            bnbh = y2-y1
            id = output[4]

            mid_bnbx = int(x1 + (bnbw / 2))
            bot_bnby = int(y1 + bnbh)
            tmp_roiid = self.rois.check_rois((mid_bnbx, bot_bnby))
            if tmp_roiid == None:
                tmp_roiid = -1


            if id in self.track_id:
                tmp_array = np.array((self.track_buffer))
                idx = [i for i, x in enumerate(tmp_array[:,0] == id) if x][0]
                cur_tracked_flag = self.track_buffer[idx][-1]
                tmp_track = [id, x1, y1, bnbw, bnbh, tmp_roiid, 0, cur_tracked_flag]
                self.track_buffer[idx] = tmp_track


            else:
                tmp_track = [id, x1, y1, bnbw, bnbh, tmp_roiid]
                # new track ID, need to add to self.track_buffer
                self.track_id.append(id)
                tmp_track.append(0)  # missed_counter
                tmp_track.append(0)  # tracked_flag
                self.track_buffer.append(tmp_track)


    def _get_priority_track_idx(self, roi_priority):

        '''

        Parameters
        ----------
        roi_priority: list of ROI rank in priority where 1st entry means most important
        where -1 is outside of RoIs.
        Examples: [2,1,-1]

        Returns: selected track' indexer, trackID
        -------

        '''
        tmp_tracks = np.array(self.track_buffer)
        tmp_tracks = np.hstack((np.array([list(range(len(tmp_tracks)))]).T,
                                 tmp_tracks))
        for i in roi_priority:
            ids_in_roi = tmp_tracks[tmp_tracks[:, 6] == i]
            if ids_in_roi.shape[0] == 0:
                continue
            else:
                for j in ids_in_roi:
                    track_state = j[-1]
                    # check if track has already been targeted zoom tracked.
                    if track_state == 0:
                        idx = j[0]
                        selected_track = tmp_tracks[int(idx), :]
                        selected_id = selected_track[1]
                        return idx, selected_id
                    else:
                        continue
        return None, None


    def init_tzt_onID(self, idx, trackid, trackduration, ptz_fps=25):
        '''

        Parameters
        ----------
        idx: indexer to self.track_buffer with the person's info to track
        trackid
        trackduration: duration in frame (exp: 3sec @ 25fps = 75frames)
        ptz_fps: look at the PTZ setting defualt is 25fps

        Returns
        -------

        '''
        if self.track_buffer[idx][0] == trackid and self.tztracking_state == False:
            self.current_tzt_id = trackid
            self.tztracking_state = True
            selected_track = self.track_buffer[idx]
            self.tzt_duration = trackduration
            track_id = int(selected_track[0])
            vid_track_name = "tzt_" + str(track_id) + ".avi"
            ptz_writer = cv2.VideoWriter(vid_track_name, cv2.VideoWriter_fourcc(*'mp4v'), ptz_fps, (2560, 1440))
            print("INIT_TZT_ID:" + str(track_id))
            return ptz_writer


    def pass_bnb(self, idx, inptz_writer):
        # Check if tzt is still within duration. if pass duration then unlock state
        # and allow for tzt of another person.
        if self.tzt_counter > self.tzt_duration or self.track_buffer[idx][6] > self.missed_track_thresh:
            self.tzt_counter = 0
            self.tztracking_state = False
            inptz_writer.release()
            selected_track = self.track_buffer[idx]
            selected_track[-1] = 1  # set the tracked flag to True, indicating this id has been tzted
            self.track_buffer[idx] = selected_track     # overwrite the record buffer
            print("DONE_TRACKING_FOR_ID: " + str(self.track_buffer[idx][0]))
            print("DONE_TRACKING_BECAUE:" + str(self.tzt_counter > self.tzt_duration) + str(self.track_buffer[idx][6] > self.missed_track_thresh))
            return None, None, None, None
        else:
            # use idx to get selected trackid from self.track_buffer
            self.tztracking_state = True
            selected_track = self.track_buffer[idx]
            selected_id = selected_track[0]
            # Check if prev_frame tzt-ID is the same as current_frame tzt-ID
            if selected_id == self.current_tzt_id:
                # the chosen tzt-ID matches prev tzt-ID and still within the tzt duration
                x1, y1, bnbw, bnbh = selected_track[1:5]
                centerx = int(x1 + (bnbw/2))
                centery = int(y1 + (bnbh/2))
                self.tzt_counter += 1
                print("TRACKING_ID: " + str(selected_id))
                return centerx, centery, bnbh, selected_id
            else:
                # if for whatever reason, selected ID doesn't match during the tzt duration
                # force the selected ID to be one from previous ID.
                self.tztracking_state = False
                self.tzt_counter = 0
                inptz_writer.release()
                end_tzt_flag = False
                print("TRACKING_ID_MISSING: " + str(selected_id))
                return None, None, None, None


    # def pass_prev_bnb(self):
    #     tmp_tracks = np.array(self.track_buffer)
    #     tmp_tracks = np.hstack((np.array([list(range(len(tmp_tracks)))]).T,
    #                             tmp_tracks))
    #     idx = int(tmp_tracks[tmp_tracks[:,1] == self.current_tzt_id][0][0])
    #     tmp_track = self.track_buffer[idx]
    #     x1, y1, bnbw, bnbh = tmp_track[1:5]
    #     centerx = int(x1 + (bnbw / 2))
    #     centery = int(y1 + (bnbh / 2))
    #     selected_id = self.current_tzt_id
    #     return centerx, centery, bnbh, selected_id


PATH_ZONE = "fixed_cam_frame.json"
TRC = TracksROIControl(PATH_ZONE)

# FIX_MTX = np.array([[1.20558436e+03, 0.00000000e+00, 9.51119590e+02],
#                 [0.00000000e+00, 1.20464537e+03, 5.38197931e+02],
#                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# FIX_DIST = np.array([[-3.92756351e-01,  1.83322215e-01,  5.99949881e-04, -2.82730083e-04, -4.42230163e-02]])
# HOMO_FIX2PTZ_LMED = np.array([[2.11118231e+00, 2.49756039e-01, -8.80940998e+02],
#                               [-2.12270102e-01, 2.05525697e+00, -3.00846856e+02],
#                               [4.02324465e-05, 6.59003849e-05, 1.00000000e+00]])


# inFixMOTtxt = "../data/Hawkeye_Dataset/dual_cam/output_primDetHawk_conf3_iou3/Fix_cam_four_people.txt"
# inFixVid = "../data/Hawkeye_Dataset/dual_cam/output_primDetHawk_conf3_iou3/Fix_cam_four_people.mp4"
# inFramePTZ = cv2.imread("PTZcam_06-09-2021.jpg")
# outVidPTZname = 'fix2ptz_tracking.avi'
# def write_fix2ptz_tracklet(inFixMOTtxt, inFixVid, inFramePTZ, outVidPTZname, inFixed_Cam_mtx, inFixed_Cam_dist, HOMO_FIX2PTZ_LMED):
#     fixtout = []
#     with open(inFixMOTtxt) as f:
#         for line in f:
#             _data = line.rstrip('\n').split(' ')[:-1]
#             _data = list(map(lambda x: float(x), _data))
#             fixtout.append(_data)
#     fixtout = np.array(fixtout)
#     fixtout_x = fixtout[:,2] + fixtout[:,4]/2
#     fixtout_y = fixtout[:,3] + fixtout[:,5]
#
#     ptz_pts = []
#     for x, y in zip(fixtout_x,fixtout_y):
#         manual_pts_fixed_undistort = cv.undistortPoints((x, y), inFixed_Cam_mtx, inFixed_Cam_dist, None, inFixed_Cam_mtx)
#         x, y = np.squeeze(manual_pts_fixed_undistort)
#         newx,newy = fixed2ptz(x, y, HOMO_FIX2PTZ_LMED)
#         ptz_pts.append((int(newx),int(newy)))
#     ptz_pts = np.array(ptz_pts)
#     fixtout = np.hstack((fixtout, ptz_pts))
#
#     vid = cv2.VideoCapture(inFixVid)
#     vid_fps = vid.get(5)
#     vid_totframe = int(vid.get(7))
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     vidwrite = cv2.VideoWriter(outVidPTZname, fourcc, vid_fps, (inFramePTZ.shape[1], inFramePTZ.shape[0]))
#     for frameiter in range(vid_totframe):
#         print(frameiter)
#         if frameiter in fixtout[:, 0]:
#             tmp_frame = np.copy(inFramePTZ)
#             data_row = fixtout[fixtout[:,0] == frameiter]
#             for id_row in data_row:
#                 id_color = compute_color_for_id(id_row[1])
#                 x = id_row[-2]
#                 y = id_row[-1]
#                 if y > inFramePTZ.shape[0] or x > inFramePTZ.shape[1]:
#                     continue
#                 else:
#                     tmp_frame = cv2.circle(tmp_frame,(int(x),int(y)), 17, id_color, thickness=-1)
#             vidwrite.write(tmp_frame)
#         else:
#             vidwrite.write(inFramePTZ)
#     vidwrite.release()
#
# write_fix2ptz_tracklet(inFixMOTtxt, inFixVid, inFramePTZ, outVidPTZname)

def fixed2ptz(x,y,inH):
    pt_1 = np.array([[x,y,1]])
    pt2_tmp = np.dot(inH, pt_1.T)
    x2 = pt2_tmp[0] / pt2_tmp[2]
    y2 = pt2_tmp[1] / pt2_tmp[2]
    return x2[0], y2[0]

################ DEBUG #############################
# fixtout = []
# with open(inFixMOTtxt) as f:
#     for line in f:
#         _data = line.rstrip('\n').split(' ')[:-1]
#         _data = list(map(lambda x: float(x), _data))
#         fixtout.append(_data)
# fixtout = np.array(fixtout)
#
# # for fi in range(int(fixtout[-1,0])):
# for fi in range(2000,6781):
#     if fi in fixtout[:,0]:
#         frame_pred = fixtout[fixtout[:, 0] == fi]
#         id = frame_pred[:, 1]
#         x1 = frame_pred[:, 2]
#         y1 = frame_pred[:, 3]
#         w = frame_pred[:,4]
#         h = frame_pred[:,5]
#         x2 = x1 + w
#         y2 = y1 + h
#         outputs_v = np.vstack((x1, y1, x2, y2, id)).T
#         TRC._update_tracks(outputs_v)
#         if TRC.tztracking_state == False:
#             sel_idx, sel_id = TRC._get_priority_track_idx([2,1,0,-1])
#
#
#         if sel_idx != None and TRC.tztracking_state == False:
#             ptz_writer = TRC.init_tzt_onID(int(sel_idx), int(sel_id), 100, 25)
#         elif sel_idx != None and TRC.tztracking_state == True:
#             centerx, centery, bnbh, selected_id = TRC.pass_bnb(int(sel_idx), ptz_writer)
#             if centerx == None:
#                 continue
#             else:
#                 manual_pts_fixed_undistort = cv.undistortPoints((centerx, centery), Fixed_Cam_mtx, Fixed_Cam_dist, None, Fixed_Cam_mtx)
#                 x, y = np.squeeze(manual_pts_fixed_undistort)
#                 ptx, pty = fixed2ptz(centerx, centery, HOMO_FIX2PTZ_LMED)
#                 mag = ptz_cam.calc_mag(bnbh, 720)
#                 p, t, z = ptz_cam.target_zoom((ptx, pty), mag)
#                 ptz_cam.absmove(p, t, z)
#                 time.sleep(0.05)
#
#             ret, ptzframe = ptz_cam.cap.read()
#             if ret and TRC.tztracking_state == True:
#                 ptz_writer.write(ptzframe)

################ DEBUG ############################


class Opt:
    def __init__(self):
        self.output = 'inference/output'
        self.source = "rtsp://admin:Hikvisionarv1234@192.168.1.65"
        # self.source = "../data/Hawkeye_Dataset/tracking/video1/SynxIPcam_urn-uuid-643C9869-12C593A8-001D-0000-000066334873_2020-11-21_17-23-00(1).mp4"
        self.yolo_weights = 'yolov5/weights/hawkeye_primarydet_2.pt'
        self.deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
        self.show_vid = False
        self.save_vid = True
        self.save_txt = True
        self.img_size = 640
        self.device = '0'
        self.evaluate = True
        self.conf_thres = 0.5
        self.iou_thres = 0.3
        self.fourcc = 'mp4v'
        self.classes = 0
        self.agnostic_nms = True
        self.augment = True
        self.config_deepsort = 'deep_sort_pytorch/configs/deep_sort.yaml'

opt = Opt()

with torch.no_grad():
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs, outputs_v = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)

                ###### PTZ ######
                TRC._update_tracks(outputs_v)
                if TRC.tztracking_state == False:
                    sel_idx, sel_id = TRC._get_priority_track_idx([2, 1, 0, -1])

                if sel_idx != None and TRC.tztracking_state == False:
                    ptz_writer = TRC.init_tzt_onID(int(sel_idx), int(sel_id), 300, 25)
                elif sel_idx != None and TRC.tztracking_state == True:
                    centerx, centery, bnbh, selected_id = TRC.pass_bnb(int(sel_idx), ptz_writer)
                    if centerx == None:
                        continue
                    else:
                        manual_pts_fixed_undistort = cv.undistortPoints((x, y), Fixed_Cam_mtx, Fixed_Cam_dist, None,
                                                                        Fixed_Cam_mtx)
                        x, y = np.squeeze(manual_pts_fixed_undistort)
                        ptx, pty = fixed2ptz(centerx, centery, HOMO_FIX2PTZ_LMED)
                        mag = ptz_cam.calc_mag(bnbh, 720)
                        p, t, z = ptz_cam.target_zoom((ptx, pty), mag)
                        ptz_cam.absmove(p, t, z)

                    ret, ptzframe = ptz_cam.cap.read()
                    if ret and TRC.tztracking_state == True:
                        ptz_writer.write(ptzframe)
                ###### PTZ ######


                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs_v, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        vx = output[6]
                        vy = output[7]
                        vr = output[8]
                        vh = output[9]
                        stationary = output[10]


                        c = int(cls)  # integer class
                        label = f'{id} {conf:.2f}'
                        color = compute_color_for_id(id)
                        plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 12 + '\n') % (frame_idx, id, cls, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, vx, vy, vr,
                                                               vh, stationary))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Draw zones boundry and state
            im0 = TRC.rois.draw_roi(im0)

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

        # ###### PTZ ######
        # TRC._update_tracks(outputs_v)
        # if TRC.tztracking_state == False:
        #     sel_idx, sel_id = TRC._get_priority_track_idx([2,1,0,-1])
        #
        # if sel_idx != None and TRC.tztracking_state == False:
        #     ptz_writer = TRC.init_tzt_onID(int(sel_idx), int(sel_id), 300, 25)
        # elif sel_idx != None and TRC.tztracking_state == True:
        #     centerx, centery, bnbh, selected_id = TRC.pass_bnb(int(sel_idx), ptz_writer)
        #     if centerx == None:
        #         continue
        #     else:
        #         manual_pts_fixed_undistort = cv.undistortPoints((x, y), Fixed_Cam_mtx, Fixed_Cam_dist, None, Fixed_Cam_mtx)
        #         x, y = np.squeeze(manual_pts_fixed_undistort)
        #         ptx, pty = fixed2ptz(centerx, centery, HOMO_FIX2PTZ_LMED)
        #         mag = ptz_cam.calc_mag(bnbh, 720)
        #         p, t, z = ptz_cam.target_zoom((ptx, pty), mag)
        #         ptz_cam.absmove(p, t, z)
        #
        #
        #     ret, ptzframe = ptz_cam.cap.read()
        #     if ret and TRC.tztracking_state == True:
        #         ptz_writer.write(ptzframe)
        # ###### PTZ ######


    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))