import numpy as np
import torch
from torch import optim

from load_data import MeanProbExtractor_yolov5, \
    MeanProbExtractor_yolov2, MeanProbExtractor_yolov8, MaxProbExtractor_yolov5
from models.common import DetectMultiBackend

dota_v1_5 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
             'ship', 'tennis-court',
             'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
             'helicopter', 'container-crane']
sandtable = ['military_vehicle', 'tank', 'warship', 'fighter_aircraft', 'carrier-based_aircraft', 'civil_aircraft',
             'barracks']
dota_sandtable = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
                  'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter',
                  'roundabout', 'soccer-ball-field', 'swimming-pool', 'military_vehicle', 'tank', 'warship',
                  'fighter_aircraft', 'carrier-based_aircraft', 'civil_aircraft', 'barrackss']
coco_80 = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic', 'light',
           'fire', 'hydrant', 'stop', 'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
           'frisbee', 'skis', 'snowboard', 'sports', 'ball', 'kite', 'baseball', 'bat', 'baseball', 'glove',
           'skateboard', 'surfboard', 'tennis', 'racket', 'bottle', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza', 'donut',
           'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell', 'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush']


class BaseConfig:
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "dataset/inria/Train/pos"
        self.val_img_dir = "dataset/inria/Test/pos"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 512
        self.start_learning_rate = 1e-3
        self.seed = 1176426343
        self.img_size = 416
        self.imgsz = (416, 416)
        self.num_classes = 80
        self.max_det = 300
        self.cls_id = 0
        self.class_names = coco_80
        self.scale = 0.2
        self.minangle = -15
        self.maxangle = 15
        self.min_brightness = -0.2
        self.max_brightness = 0.2
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.noise_factor = 0.1
        self.offsetx = 0.02
        self.offsety = 0.05
        self.by_rect = True
        self.rand_loc = False

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        self.patch_name = 'base'
        self.device = torch.device('cuda:0')
        self.dtype = torch.float32

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50, verbose=True)
        self.max_tv = 0

        self.batch_size = 16

        self.loss_target = lambda obj, cls: obj * cls  # self.loss_target(obj, cls) return obj * cls

        self.generator = torch.Generator(self.device).manual_seed(self.seed)

        self.init_num_inference_steps = 50
        self.num_inference_steps = 4
        self.start_time_step = 601
        self.end_time_step = 1
        self.do_classifier_free_guidance = True
        self.guidance_scale = 7
        # self.prompt = ['circular dragon logo, full body, masterpiece, high quality, 8k']
        # self.prompt = ['1golden retriever, portrait, extremely detailed, masterpiece, high quality, 8K']
        self.prompt = ['1Pomeranian, portrait, extremely detailed, masterpiece, high quality, 8K']
        self.negative_prompt = [
            'low quality, bad anatomy, wrong anatomy, deformed iris, deformed pupils, distorted, disfigured, '
            'watermark, text, signature']


class yolov2(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolo.cfg"
        self.weights = "weights/yolov2.weights"
        self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov2"
        self.mode = 'yolov2'
        self.patch_name = 'yolov2'
        self.max_tv = 0.165
        self.batch_size = 16

        self.loss_target = lambda obj, cls: obj
        from darknetv2 import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov2(self.cls_id, self.model.num_classes,
                                                       self.model.num_anchors,
                                                       self.model.anchors,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)


class yolov3(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov3.cfg"
        self.weights = "weights/yolov3.weights"
        self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov3"
        self.val_lab_dir = "dataset/inria/Test/pos/yolo-labels_yolov3"
        self.mode = 'yolov5'
        self.patch_name = 'yolov3'
        self.max_tv = 0.165
        self.batch_size = 16

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image

        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)


class yolov3_dota(BaseConfig):
    def __init__(self):
        super().__init__()
        self.weights = "weights/yolov3_dotasp.pt"
        self.img_dir = '../datasets/DOTA_SP/train/images/'
        self.lab_dir = "../datasets/DOTA_SP/train/labels_yolov3_dota"
        self.val_img_dir = '../datasets/DOTA_SP/val/images/'
        self.val_lab_dir = '../datasets/DOTA_SP/val/labels_yolov3_dota/'
        self.mode = 'yolov5'
        self.patch_name = 'yolov3_dota'
        self.max_tv = 0.165
        self.batch_size = 2
        self.loss_target = lambda obj, cls: obj
        self.num_classes = 16
        self.imgsz = (1024, 1024)
        self.img_size = 1024
        self.scale = 0.1
        self.cls_id = 0
        self.prompt = ['camouflage, sea']
        self.negative_prompt = ['']
        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)


class yolov3tiny(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov3-tiny.cfg"
        self.weights = "weights/yolov3-tiny.weights"
        self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov3tiny"
        self.mode = 'yolov5'
        self.patch_name = 'yolov3tiny'
        self.max_tv = 0.165
        self.batch_size = 24

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)


class yolov3tiny_mpii(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov3-tiny.cfg"
        self.weights = "weights/yolov3-tiny.weights"
        self.img_dir = 'dataset/mpii/train'
        self.lab_dir = "dataset/mpii/train/labels_yolov3tiny"
        self.val_img_dir = 'dataset/mpii/test'
        self.mode = 'yolov5'
        self.patch_name = 'yolov3tiny-mpii'
        self.max_tv = 0.165
        self.batch_size = 24

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.6  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)


class yolov3tiny_mix(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov3-tiny.cfg"
        self.weights = "weights/yolov3-tiny.weights"
        self.img_dir = 'dataset/mix/train'
        self.lab_dir = "dataset/mix/train/labels_yolov3tiny"
        self.val_img_dir = 'dataset/mix/test'
        self.mode = 'yolov5'
        self.patch_name = 'yolov3tiny-mix'
        self.max_tv = 0.165
        self.batch_size = 24

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)


class yolov4(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov4.cfg"
        self.weights = "weights/yolov4.weights"
        self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov4"
        self.mode = 'yolov5'
        self.patch_name = 'yolov4'
        self.max_tv = 0.165
        self.batch_size = 16

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)


class yolov4tiny(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov4-tiny.cfg"
        self.weights = "weights/yolov4-tiny.weights"
        self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov4tiny"
        self.mode = 'yolov5'
        self.patch_name = 'yolov4tiny'
        self.max_tv = 0.165
        self.batch_size = 16

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)


class yolov5s(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov5s'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = 'weights/yolov5s-416.pt'
        self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov5s"

        self.mode = 'yolov5'
        self.imgsz = (416, 416)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)


class yolov5s_st(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov5s_st'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = 'weights/yolov5s_st.pt'
        self.img_dir = 'dataset/fushi/train/images'
        self.lab_dir = "dataset/fushi/train/labels_yolov5s_st"
        self.val_img_dir = 'dataset/fushi/val/images'
        self.val_lab_dir = "dataset/fushi/val/labels_yolov5s_st"

        self.prompt = ['camouflage, sea']
        self.negative_prompt = ['']

        self.scale = 0.2
        self.minangle = -45
        self.maxangle = 45
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.noise_factor = 0.25
        self.offsetx = 0.05
        self.offsety = 0.05
        self.rand_loc = True
        self.by_rect = False

        self.num_inference_steps = 3
        self.guidance_scale = 5

        self.class_names = sandtable
        self.mode = 'yolov5'
        self.num_classes = 7
        self.cls_id = 3
        self.batch_size = 16
        self.imgsz = (640, 640)
        self.img_size = 640
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)


class yolov5s_dotast(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov5s_dotast'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = 'weights/yolov5s_dotast.pt'
        self.img_dir = 'dataset/fushi/train/images'
        self.lab_dir = "dataset/fushi/train/labels_yolov5s_dotast"
        self.val_img_dir = 'dataset/fushi/val/images'
        self.val_lab_dir = "dataset/fushi/val/labels_yolov5s_dotast"

        self.prompt = ['camouflage, sea']
        self.negative_prompt = ['']

        self.scale = 0.2
        self.minangle = -45
        self.maxangle = 45
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.noise_factor = 0.25
        self.offsetx = 0.05
        self.offsety = 0.05
        self.rand_loc = True
        self.by_rect = False

        self.num_inference_steps = 3
        self.guidance_scale = 5
        self.class_names = dota_sandtable
        self.mode = 'yolov5'
        self.num_classes = 22
        self.cls_id = 18
        self.batch_size = 16
        self.imgsz = (640, 640)
        self.img_size = 640
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)


class yolov8s(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov8s'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = 'weights/yolov8s-416.pt'
        self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov8s"

        self.mode = 'yolov8'
        self.imgsz = (416, 416)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MeanProbExtractor_yolov8(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)


patch_configs = {
    "yolov2": yolov2,
    "yolov3": yolov3,
    "yolov3_dota": yolov3_dota,
    "yolov3tiny": yolov3tiny,
    "yolov3tiny-mpii": yolov3tiny_mpii,
    "yolov3tiny-mix": yolov3tiny_mix,
    "yolov4": yolov4,
    "yolov4tiny": yolov4tiny,
    "yolov5s": yolov5s,
    "yolov5s_st": yolov5s_st,
    "yolov5s_dotast": yolov5s_dotast,
    "yolov8s": yolov8s,
}
