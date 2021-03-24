from lvis import LVIS
import numpy as np
import pickle
import pdb
import os
import json
import torch
from pycocotools.coco import COCO

train_ann_file = "/root/data/zq/data/SMD/annotations/6c/SMD_VIS_6_class_train_Qianna.json"
with open(train_ann_file, 'r') as f:
    gt_new = json.load(f)
c = [0, 0, 0, 0, 0, 0]
anns = gt_new["annotations"]
for ann in anns:
    if ann['category_id'] == 1:
        c[0] += 1
    elif ann['category_id'] == 2:
        c[1] += 1
    elif ann['category_id'] == 3:
        c[2] += 1
    elif ann['category_id'] == 4:
        c[3] += 1
    elif ann['category_id'] == 5:
        c[4] += 1
    else:
        c[5] += 1
print(c)
categories_new = []
categories = gt_new["categories"]
for i, cate in enumerate(categories):
    cate["instance_count"] = c[i]
    categories_new.append(cate)
gt_new["categories"] = categories_new
with open("/root/data/zq/data/SMD/lvis/SMD_VIS_6_class_train_Qianna_lvis.json", 'w') as g:
    json.dump(gt_new, g)
lvis_ann_file = "/root/data/zq/data/SMD/lvis/SMD_VIS_6_class_train_Qianna_lvis.json"
# test_ann_file = "/root/data/zq/data/SMD/annotations/6c/SMD_VIS_6_class_test.json"

lvis_train = LVIS(lvis_ann_file)
train_catsinfo = lvis_train.cats

binlabel_count = [1, 1, 1, 1]
label2binlabel = np.zeros((4, 6), dtype=np.int)

label2binlabel[0, :-1] = binlabel_count[0]
binlabel_count[0] += 1

for cid, cate in train_catsinfo.items():
    print(cid, cate)
    ins_count = cate['instance_count']
    if ins_count < 1000:
        label2binlabel[1, cid-1] = binlabel_count[1]
        binlabel_count[1] += 1
    elif ins_count < 10000:
        label2binlabel[2, cid-1] = binlabel_count[2]
        binlabel_count[2] += 1
    else:
        label2binlabel[3, cid-1] = binlabel_count[3]
        binlabel_count[3] += 1


savebin = torch.from_numpy(label2binlabel)

save_path = '/root/data/zq/data/SMD/lvis/label2binlabel.pt'
torch.save(savebin, save_path)

# start and length
pred_slice = np.zeros((4, 2), dtype=np.int)
start_idx = 0
for i, bincount in enumerate(binlabel_count):
    pred_slice[i, 0] = start_idx
    pred_slice[i, 1] = bincount
    start_idx += bincount

savebin = torch.from_numpy(pred_slice)
save_path = '/root/data/zq/data/SMD/lvis/pred_slice_with0.pt'
torch.save(savebin, save_path)

# for tarining set
# lvis_train = LVIS(train_ann_file)
# lvis_val = LVIS(val_ann_file)
# train_catsinfo = lvis_train.cats
# val_catsinfo = lvis_val.cats

bin1000 = []
bin10000 = []
binover = []

for cid, cate in train_catsinfo.items():
    ins_count = cate['instance_count']
    if ins_count < 1000:
        bin1000.append(cid)
    elif ins_count < 10000:
        bin10000.append(cid)
    else:
        binover.append(cid)

splits = {}
# splits['(0, 10)'] = np.array(bin10, dtype=np.int)
# splits['[10, 100)'] = np.array(bin100, dtype=np.int)
splits['[100, 1000)'] = np.array(bin1000, dtype=np.int)
splits['[1000, 10000)'] = np.array(bin10000, dtype=np.int)
splits['[10000, ~)'] = np.array(binover, dtype=np.int)
splits['normal'] = np.arange(6)
splits['background'] = int(6)
splits['all'] = np.arange(7)

split_file_name = '/root/data/zq/data/SMD/lvis/valsplit.pkl'
with open(split_file_name, 'wb') as f:
    pickle.dump(splits, f)
