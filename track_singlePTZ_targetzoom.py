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
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from zone import Zone, Zones
from sensecam_control import onvif_control
from matplotlib import pyplot as plt
from Utils import cap_frame, see_center
from zone import draw_zone_on_frame
import pickle as pkl
from target_zoom import *

RTSP = "rtsp://admin:Hikvisionarv1234@192.168.1.64"
ip = '192.168.1.64'
login = 'arvonvif'
password = 'Arvonvif1234'
cam = onvif_control.CameraControl(ip, login, password)
cam.camera_start()
PATH_ZONE = "zone_frame.json"
# PATH_ZONE = "../data/Hawkeye_Dataset/stationary_tracker/SynxIPcam_2020-11-20_10-41-59.70.json"
Zones = Zones(PATH_ZONE)
with open("PTZ_calibration.pkl", 'rb') as f:
    cam_param = pkl.load(f)
    MTX = cam_param['mtx']
with open("PTZ_P_Matrix.pkl", 'rb') as f:
    PTZ_P_Matrix = pkl.load(f)
    P_Matrix = PTZ_P_Matrix['P_Matrix']

ptz1 = PTZ(ip, login, password, P_Matrix, MTX)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


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
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


class Opt:
    def __init__(self):
        self.output = 'inference/output'
        # self.source = "rtsp://admin:Hikvisionarv1234@192.168.1.64"
        self.source = "192.168.1.64_01_2021082816381881.mp4"
        # self.source = "../data/Hawkeye_Dataset/tracking/video1/SynxIPcam_urn-uuid-643C9869-12C593A8-001D-0000-000066334873_2020-11-21_17-23-00(1).mp4"
        self.yolo_weights = 'yolov5/weights/hawkeye_primarydet_2.pt'
        self.deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
        self.show_vid = False
        self.save_vid = False
        self.save_txt = True
        self.img_size = 640
        self.device = 'cpu'
        self.evaluate = True
        self.conf_thres = 0.5
        self.iou_thres = 0.3
        self.fourcc = 'mp4v'
        self.classes = 0,1
        self.agnostic_nms = True
        self.augment = True
        self.config_deepsort = 'deep_sort_pytorch/configs/deep_sort.yaml'
        self.ip = '192.168.1.64'
        self.login = 'arvonvif'
        self.password = 'Arvonvif1234'
        self.zones = "../data/Hawkeye_Dataset/stationary_tracker/SynxIPcam_2020-11-20_10-41-59.70.json"

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

        for zone in Zones.zonelist:
            zone._reset()

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

                        # Only associate cls 0 (person) who is stationary to Zones
                        if stationary == 0 and cls == 0:
                            still_wrt = 'o'
                        else:
                            print("STATIONARY HERE\n")
                            still_wrt = '*'
                            x1, y1, x2, y2 = output[0:4]
                            xmid_feet = int(output[0] + ((output[2] - output[0])/2))
                            ybot_feet = int(output[1] + (output[3] - output[1]))

                            tmp_zoneidx = Zones.check_zone((xmid_feet, ybot_feet))
                            if tmp_zoneidx is not None:
                                tmp_zone = Zones.zonelist[tmp_zoneidx]
                                tmp_zoneid = tmp_zone.zoneID
                                tmp_zone._add_still(bboxes)


                        c = int(cls)  # integer class
                        label = f'{id} {still_wrt} {conf:.2f}'
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


        # Update Zonelist
        # Zones update number of still in each zone
        zone_stillcount = Zones.update_zone_still_count()

        # Decision to pick which zone or person to target zoom
        def pick_zone_tz(in_zonelist_buffer):
            # THRESH_STILL is how many consecutive nstill in a zone per frame to trigger zoom
            # This threshold is equivalent to 1 person standing still in a zone for at least 2sec
            # at 20fps totaling 40. The mean of this specific zone will buffer size of 40 is 1.
            # This is to prevent jitter noise of false stationary bnb
            THRESH_STILL = 0.1
            tmp_zonelist_buffer = np.array(in_zonelist_buffer)
            meanA = np.mean(tmp_zonelist_buffer, axis=0)
            meanA_sort_idx = np.argsort(meanA)[::-1]
            for i in meanA_sort_idx:
                if Zones.zones_tz_rec[i] == False and meanA[i] > THRESH_STILL:
                    return i, meanA
            return None, meanA

        idx_zone_tz, meanA = pick_zone_tz(Zones.zonelist_buffer)
        print(idx_zone_tz)
        meanA = list(meanA)
        with open("zonelog.txt", 'a') as f:
            f.write(str(meanA)[1:-1] + '\n') # label format

        if idx_zone_tz != None:
            tmp_zone = Zones.zonelist[idx_zone_tz]
            print("TARGET ZOOM HERE at ", str(tmp_zone.zoneID))
            ptx, pty, bnbh = tmp_zone.get_bnb_pt_tz()
            mag = ptz1.calc_mag(bnbh, 720)
            p, t, z = ptz1.target_zoom((ptx, pty), mag)
            ptz1.absmove(p, t, z)
            target_zooming = True
            while target_zooming:
                # During target zooming, tracking will be temporary pause
                # snapshots of frame will be taken
                time.sleep(5)
                target_zooming = False
                print("Done TARGET ZOOM!!!!")
                Zones.zones_tz_rec[idx_zone_tz] = True

            ptz1.absmove(*ptz1.home_ptx)
        Zones.update_frame_count()








    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
#     parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
#     # file/folder, 0 for webcam
#     parser.add_argument('--source', type=str, default='0', help='source')
#     parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
#     parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
#     parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
#     parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
#     # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--evaluate', action='store_true', help='augmented inference')
#     parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
#     args = parser.parse_args()
#     args.img_size = check_img_size(args.img_size)
#
#     with torch.no_grad():
#         detect(args)


_, homeframe = cap_frame(RTSP)
for i, zone in enumerate(Zones.zonelist):
    if i == 3:
        color = (0,0,255)
    else:
        color = (0,255,0)
    homeframe = draw_zone_on_frame(homeframe, zone, color)
cv2.imwrite("zone_home_frame.jpg", homeframe)

with open("output/infernece/vee_Test.txt", 'a') as f:
    f.write("test" + '\n')
