import sys
sys.path.insert(0, './yolov5')
from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams, LoadStreams_2
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow,xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5.utils.plots import plot_one_box
import os
import platform
import shutil
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from zone import ROIS
from collections import deque
import yaml
import threading
import queue
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
        if len(self.track_buffer) == 0:
            return None, None
        else:
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


    def init_tzt_onID(self, idx, trackid, trackduration, vidpathwrite, ptz_fps=25):
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
            vid_track_name = vidpathwrite + "_tzt_" + str(track_id) + ".mp4"
            ptz_writer = cv2.VideoWriter(vid_track_name, cv2.VideoWriter_fourcc(*'mp4v'), ptz_fps, (2560, 1440))
            print("INIT_TZT_ID:" + str(track_id))
            print(ptz_writer)
            return ptz_writer


    def pass_bnb(self, idx, inptz_writer):
        # Check if tzt is still within duration. if pass duration then unlock state
        # and allow for tzt of another person.
        if self.tzt_counter > self.tzt_duration or self.track_buffer[idx][6] > self.missed_track_thresh:
            print("DONE_TRACKING_FOR_ID: " + str(self.track_buffer[idx][0]))
            print("DONE_TRACKING_BECAUSE:" + str(self.tzt_counter > self.tzt_duration) + str(self.track_buffer[idx][6] > self.missed_track_thresh))
            self.tzt_counter = 0
            self.tztracking_state = False
            inptz_writer.release()
            selected_track = self.track_buffer[idx]
            selected_track[-1] = 1  # set the tracked flag to True, indicating this id has been tzted
            self.track_buffer[idx] = selected_track     # overwrite the record buffer
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
                print("TRACKING_ID_MISSING: " + str(selected_id))
                return None, None, None, None


PATH_ZONE = "fixed_cam_frame.json"
TRC = TracksROIControl(PATH_ZONE)

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    if self.q.empty():
      return None
    else:
      return self.q.get()


def fixed2ptz(x,y,inH):
    pt_1 = np.array([[x,y,1]])
    pt2_tmp = np.dot(inH, pt_1.T)
    x2 = pt2_tmp[0] / pt2_tmp[2]
    y2 = pt2_tmp[1] / pt2_tmp[2]
    return x2[0], y2[0]



class Opt:
    def __init__(self):
        self.output = 'inference/output'
        self.source = "rtsp://admin:Hikvisionarv1234@192.168.1.65"
        self.yolo_weights = 'yolov5/weights/yolov5s_hawkeye.pt'
        self.deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
        self.show_vid = True
        self.save_vid = True
        self.save_txt = True
        self.img_size = 640
        self.device = 'cpu'
        self.evaluate = True
        self.conf_thres = 0.6
        self.iou_thres = 0.3
        self.fourcc = 'mp4v'
        self.classes = 1
        self.agnostic_nms = True
        self.augment = True
        self.config_deepsort = 'deep_sort_pytorch/configs/deep_sort.yaml'

opt = Opt()

ptz_vidcap = VideoCapture('rtsp://admin:Hikvisionarv1234@192.168.1.64')
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
        dataset2 = LoadStreams_2(source, img_size=imgsz, stride=stride)
        #dataset = LoadStreams(source, img_size=imgsz, stride=stride)
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

    # frame_idx2, (path2, img2, im0s2, vid_cap2) = next(iter(enumerate(dataset2)))
    # frame_idx, (path, img, im0s, vid_cap) = next(iter(enumerate(dataset)))

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset2):
        if frame_idx >= 120:
            vid_writer.release()
            break
        else:
            path = [path]
            im0s = [im0s]
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
                save_path_ptz = str(Path(out) / Path(p).name)

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
                        ptz_writer = TRC.init_tzt_onID(int(sel_idx), int(sel_id), 30, save_path_ptz, 6)
                    elif sel_idx != None and TRC.tztracking_state == True:
                        centerx, centery, bnbh, selected_id = TRC.pass_bnb(int(sel_idx), ptz_writer)
                        if centerx == None:
                            continue
                        else:
                            manual_pts_fixed_undistort = cv.undistortPoints((centerx, centery), Fixed_Cam_mtx, Fixed_Cam_dist, None, Fixed_Cam_mtx)
                            x, y = np.squeeze(manual_pts_fixed_undistort)
                            ptx, pty = fixed2ptz(x, y, HOMO_FIX2PTZ_LMED)
                            mag = ptz_cam.calc_mag(bnbh, 720)
                            pan, tilt, zoom = ptz_cam.target_zoom((ptx, pty), mag)
                            ptz_cam.absmove(pan, tilt, zoom)
                        imgptz = ptz_vidcap.read()
                        if imgptz is None:
                            pass
                        else:
                            ptz_writer.write(imgptz)
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
                            fps, w, h = 6, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
