import gradio as gr
import torch
import torchvision.transforms as T

import patch_config
from load_data import preds2boxes
from post_util import plot_boxes

cfg = patch_config.patch_configs['yolov3']()
model = cfg.model
model.eval()

css = ".output-image, .input-image, .image-preview {height: 600px !important} "


def detect_objects(image):
    torch.cuda.empty_cache()
    trans = T.Compose([
        T.Resize(cfg.imgsz),
        T.ToTensor()
    ])
    img = trans(image)
    img = img.unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(img)
        boxes = preds2boxes(cfg, output)
    result = plot_boxes(image, boxes[0], class_names=cfg.class_names)
    return result


input_component = gr.Image(label="拍照", sources=['webcam', 'upload', 'clipboard'], type='pil')
output_component = gr.Image(label="检测结果", show_download_button=True)

gr.Interface(
    fn=detect_objects,
    inputs=input_component,
    outputs=output_component,
    title="目标检测",
    description="通过摄像头拍照并检测图像中的目标。",
).launch(server_name='0.0.0.0', server_port=18000, debug=True, ssl_verify=False)
