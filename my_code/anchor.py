import glob
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import json
from sklearn.cluster import KMeans

annotationFile = '/root/vsislab-2/zq/data/IKCEST3rd_bbox_detection/annotations/ikcest_train_bbox_annotations.json'
# annotationFile = ''

num_clusters = 9

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
for i, ann in enumerate(annotations['annotations']):
    bbox = ann['bbox']
    bbox_list.append(bbox[2:])
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
