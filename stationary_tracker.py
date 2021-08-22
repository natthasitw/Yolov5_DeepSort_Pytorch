import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint
from zone import Zone
from zone import draw_zone_on_frame
import pandas as pd
import sys
import os


PATH_ZONE = "../data/Hawkeye_Dataset/stationary_tracker/SynxIPcam_2020-11-20_10-41-59.70.json"
PATH_IMG = "../data/Hawkeye_Dataset/stationary_tracker/SynxIPcam_2020-11-20_10-41-59.70.bmp"
INPUT_VID = "../data/Hawkeye_Dataset/tracking/video1/SynxIPcam_urn-uuid-643C9869-12C593A8-001D-0000-000066334873_2020-11-21_17-23-00(1).mp4"
img = cv2.cvtColor(cv2.imread(PATH_IMG), cv2.COLOR_BGR2RGB)
PATH_INFERENCE = "inference/output/SynxIPcam_urn-uuid-643C9869-12C593A8-001D-0000-000066334873_2020-11-21_17-23-00(1).txt"


cap = cv2.VideoCapture(INPUT_VID)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with open(PATH_ZONE) as f:
  data = json.load(f)

label = []
with open(PATH_INFERENCE) as f:
    for line in f:
        _data = line.rstrip('\n').split(' ')[:-1]
        _data = list(map(lambda x: float(x), _data))
        label.append(_data)
label = np.array(label)

df = pd.DataFrame(label, dtype=float, columns=['frame', 'id', 'cls', 'x_left', 'y_top', 'width', 'height', 'vx', 'vy', 'vr', 'vh','stationary'])
frame_pred = df.loc[df['frame'] == 670, ['id', 'cls', 'x_left', 'y_top', 'width', 'height','vx', 'vy', 'vr', 'vh']].values

low_motion = df[(df.vx < 0.1) & (df.vx > -0.1) & (df.vy < 0.1) & (df.vy > -0.1) & (df.vh < 0.1) & (df.vh > -0.1)]
A23 = df[(df.vx < 0.1) & (df.vx > -0.1) & (df.vy < 0.1) & (df.vy > -0.1) & (df.vh < 0.1) & (df.vh > -0.1) & (df.id == 23)]

cap.set(1,670)
_, frame2 = cap.read()
frame3 = frame2[:,:,[2,1,0]].copy()

bnb_list = list(frame_pred[:,:5])
for bnb in bnb_list:
    _id, _x1, _y1, _x2, _y2 = bnb
    cv2.rectangle(frame3, (int(_x1), int(_y1)), (int(_x1 + _x2), int(_y1 + _y2)), color=(255,0,0), thickness=3)


zonelist = []
for i in data['shapes']:
  zone_id = i['label']
  coord = i['points']
  zone = Zone(coord, zone_id)
  zonelist.append(zone)









sample_id = A[A['frame'] == 1788].to_numpy()
x1,y1,w,h = sample_id[0,3:7:1]
centerx = int(x1 + (w/2))
bottomy = int(y1 + h)

zone_listcheck = list(map(lambda z: z.is_inside_polygon((centerx, bottomy)), zonelist))

[i for i, x in enumerate(zone_listcheck) if x]
zonelist[31].zoneID

track_id = 31
tmp_zone = zonelist[track_id]
if track_id not in tmp_zone.tzIDs:
    # target zoom in
    tmp_zone._add_ID(track_id, 0)






