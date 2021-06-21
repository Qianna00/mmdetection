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
    bbox_list.append(bbox)
bboxes = np.array(bbox_list)

kmeans = KMeans(n_clusters=num_clusters)
anchors = kmeans.fit(bboxes)

print(anchors)
