from mmdet.apis import inference_detector
from mmdet.apis import init_detector
from mmdet.apis import show_result_pyplot

config_file = "/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_c4_1x_smd.py"

checkpoint_file = "/root/data/zq/smd_det/frcnn_res50_mmdet/10c/epoch_8.pth"

model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = "/root/data/zq/smd_det/test_imgs/test/1.jpg"
result = inference_detector(model, img)
show_result_pyplot(model, img, result)
