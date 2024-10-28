import cv2
import torchvision
from PIL import Image
from torchvision import transforms
import load_data
import patch_config
import post_util
import numpy as np
import os
import torch

# 打开原始视频文件
input_video = 'dataset/fushi.avi'
v_name = os.path.splitext(os.path.basename(input_video))[0]
cap = cv2.VideoCapture(input_video)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的基本信息
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 输出视频文件的设置
output_video = f'predict_{v_name}.mp4'
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

cfg = patch_config.patch_configs['yolov3']()
model = cfg.model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(cfg.imgsz)
])

# 处理视频的每一帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(frame).unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(img)
    boxes = load_data.preds2boxes(cfg, output)
    plot_img = post_util.plot_boxes(frame, boxes[0], class_names=cfg.class_names, fill=True)
    out_img = np.array(plot_img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    out.write(out_img)
    # 将处理后的帧写入输出视频

    # 显示处理后的帧（可选）
    # cv2.imshow('Frame with Box', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
