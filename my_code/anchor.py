import glob
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import json
from sklearn.cluster import KMeans

annotationFile = '/root/vsislab-2/zq/data/IKCEST3rd_bbox_detection/annotations/ikcest_train_bbox_annotations.json'
# annotationFile = ''

data_root = "/root/vsislab-2/zq/data/SMD_MOT/MOT16_format/images/train/"

num_clusters = 12

size = [360, 640]

with open(annotationFile, 'r') as f:
    annotations = json.load(f)
"""size_list = []

for i, img in enumerate(annotations['images']):
    img_size = (img['width'], img['height'])
    if img_size not in size_list:
        size_list.append(img_size)
print(size_list)"""
# (1920,1080) (1280,720) (1920,1072) (1920,1088)

bbox_list = []
"""for i, ann in enumerate(annotations['annotations']):
    bbox = ann['bbox']
    bbox_list.append(bbox[2:])"""
for vid in os.listdir(data_root):
    vid_path = os.path.join(data_root, vid)
    gt_path = vid_path + '/gt/gt.txt'
    with open(gt_path, 'r') as f:
        gt_list = f.readlines()
    for gt in gt_list:
        bbox = gt.split(',')[4:6]
        bbox_list.append(bbox)

bboxes = np.array(bbox_list)

kmeans = KMeans(n_clusters=num_clusters)
anchors = kmeans.fit(bboxes)

print(anchors.cluster_centers_)

# [[ 819.76664478  637.74983585   53.064478     49.82744583]  53  50
#  [  92.48940533  443.25112782  119.99822283  107.5393028 ]  120 108
#  [ 628.43845922  391.9327415    43.9112248    42.28287692]  44  42
#  [1602.15443896  552.45807645  109.47163995  111.79284834]  109 112
#  [1050.33996212  407.76799242  407.05018939  321.78693182]  407 322
#  [ 916.52112866  378.93708654   66.21938707   60.60874848]  66  61
#  [ 381.35107185  386.52860067   52.03641064   47.0515997 ]  52  47
#  [1178.6986987   540.07357357   77.79362696   73.77827828]  78  74
#  [ 463.01855406  630.64619322   74.21177223   62.1399019 ]] 74  62

# [[44, 42], [52, 47], [53, 50], [66, 61], [74, 62], [78, 74], [109, 112], [120, 108], [407, 322]]

# IKCEST2021
# [[ 24.08195292  27.54004776]  24  28
#  [326.28061582 214.93212036]  326 215
#  [ 58.64028692  56.03207066]  59  56
#  [ 78.09536576 157.22481808]  78  157
#  [640.94308943 716.91056911]  641 717
#  [120.47385664  80.50163806]  120 81
#  [200.56102962 142.17077221]  201 142
#  [217.4173913  350.16521739]  217 350
#  [494.23175182 369.93978102]] 494 369

# [[24, 28], [59, 56], [78, 157], [120, 81], [201, 142], [217, 350], [326, 215], [494, 369], [641, 717]]

# SMD_ 9 anchors
# [[ 125.47108852  116.60655263]  125  117
#  [ 679.68858939  165.52933659]  680  166
#  [ 491.50929916  139.98314505]  492  140
#  [1282.80066079  319.40374449]  1283 319
#  [ 217.60342272   63.84569855]  218  64
#  [  44.8730399    30.49345535]  45   30
#  [ 147.28487778  263.01338889]  147  263
#  [ 306.04025957   97.17218689]  306  97
#  [ 119.32146215   51.27157548]] 119  51

# [[45, 30], [119, 51], [218 64], [125, 117], [306, 97], [147, 263], [492, 140], [680, 166], [1283, 319]]

