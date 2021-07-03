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
INPUT_VID = "../data/Hawkeye_Dataset/stationary_tracker/SynxIPcam_urn-uuid-643C9869-12C593A8-001D-0000-000066334873_2020-11-21_17-23-00(1).mp4"
img = cv2.cvtColor(cv2.imread(PATH_IMG), cv2.COLOR_BGR2RGB)
PATH_INFERENCE = "inference/output_2/SynxIPcam_urn-uuid-643C9869-12C593A8-001D-0000-000066334873_2020-11-21_17-23-00(1).txt"


cap = cv2.VideoCapture(INPUT_VID)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with open(PATH_ZONE) as f:
  data = json.load(f)

label = []
with open(PATH_INFERENCE) as f:
    for line in f:
        data = line.rstrip('\n').split(' ')[:-1]
        data = list(map(lambda x: int(x), data))
        label.append(data)
label = np.array(label)

df = pd.DataFrame(label, dtype=int, columns=['frame', 'id', 'x_left', 'y_top', 'width', 'height', 'N', 'N', 'N', 'N'])
B = df.loc[df['frame'] == 2, ['x_left', 'y_top', 'width', 'height']].values

cap.set(0,1)
_, frame2 = cap.read()
frame3 = frame2[:,:,[2,1,0]].copy()

start = (int(B[0][0]), int(B[0][1]))
end = (int(B[0][0] + B[0][2]), int(B[0][1] + B[0][3]))
frame3 = cv2.rectangle(frame3, start, end, (255,0,0),5)

zonelist = []
for i in data['shapes']:
  zone_id = i['label']
  coord = i['points']
  zone = Zone(coord, zone_id)
  zonelist.append(zone)












