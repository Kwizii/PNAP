import fnmatch
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from ultralytics.utils.ops import non_max_suppression as nms_y8

from median_pool import MedianPool2d
from post_util import get_region_boxes, nms
from utils.general import non_max_suppression as nms_y5, non_max_suppression_gradable
from utils.general import xyxy2xywh


class MaxProbExtractor_yolov2(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, loss_target):
        super(MaxProbExtractor_yolov2, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = loss_target

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls, h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls, 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        # confs_if_object = output_objectness  # confs_for_class * output_objectness
        # confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.loss_target(output_objectness, confs_for_class)
        # print(confs_if_object,len(confs_if_object[0]))
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)
        # print(max_conf)

        return max_conf


class MeanProbExtractor_yolov2(nn.Module):
    def __init__(self, cls_id, num_cls, num_anchors, anchors, loss_target, conf_thres, iou_thres, max_det):
        super(MeanProbExtractor_yolov2, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.num_anchors = num_anchors
        self.anchors = anchors
        self.loss_target = loss_target
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def forward(self, output):
        boxes = get_region_boxes(output, self.conf_thres, self.num_cls, self.anchors,
                                 self.num_anchors)
        conf_list = []
        for box in boxes:
            mask = box[:, -1] == self.cls_id
            tgt_box = box[mask]
            if len(tgt_box) == 0:
                continue
            conf_list.append(tgt_box[:, 4].mean())
        if len(conf_list) == 0:
            return torch.tensor(0., device=output.device, dtype=output.dtype)
        return torch.stack(conf_list)
        # for i in range(len(nms_boxes)):
        #     if len(nms_boxes[i]) == 0:
        #         continue
        #     bboxes = torch.stack([torch.stack(box) for box in nms_boxes[i]])
        #     bboxes = bboxes[bboxes[:, -1] == self.cls_id].cuda()
        #     if len(bboxes) > 0:
        #         mean_conf.append(torch.mean(bboxes[:, 4]))
        # if len(mean_conf) == 0:
        #     mean_conf.append(torch.tensor(float(0)))
        # return torch.stack(mean_conf)


class MaxProbExtractor_yolov5(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, loss_target):
        super(MaxProbExtractor_yolov5, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = loss_target

    def forward(self, YOLOoutput):  # YOLOoutput: torch.Size([batch, 64512, 85])
        if isinstance(YOLOoutput, list):
            YOLOoutput = YOLOoutput[0]
        output = YOLOoutput.transpose(1, 2).contiguous()  # [batch, 85, 64512]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 64512]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 64512]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        # confs_if_object = output_objectness  # confs_for_class * output_objectness
        # confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.loss_target(output_objectness, confs_for_class)
        # confs_if_object = output_objectness * confs_for_class
        # print(confs_if_object, len(confs_for_class[0]))
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)
        # print(max_conf)
        return max_conf


class MeanProbExtractor_yolov5(nn.Module):
    def __init__(self, cls_id, num_cls, loss_target, conf_thres, iou_thres, max_det):
        super().__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = loss_target
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def forward(self, YOLOoutput):
        if isinstance(YOLOoutput, list):  # YOLOoutput: torch.Size([batch, 64512, 85])
            YOLOoutput = YOLOoutput[0]
        conf_list = []
        YOLOoutput = non_max_suppression_gradable(YOLOoutput, conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                                  classes=self.cls_id, max_det=self.max_det)
        for box in YOLOoutput:
            if len(box) == 0:
                continue
            conf_list.append(box[:, 4].mean())
        # for box in YOLOoutput:
        #     fbox = box[box[:, 4] > self.conf_thres]
        #     i = torch.argmax(fbox[:, 5:], dim=1)
        #     mask = i == self.cls_id
        #     tgt_box = fbox[mask]
        #     if len(tgt_box) == 0:
        #         continue
        #     conf_list.append(tgt_box[:, 4].mean())
        # bpreds = nms_gradable(YOLOoutput, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=self.cls_id)
        # for preds in bpreds:
        #     if len(preds) == 0:
        #         conf_list.append(torch.tensor(0., device=preds.device, dtype=preds.dtype))
        #         continue
        #     conf_list.append(preds[:, 4].mean())
        if len(conf_list) == 0:
            return torch.tensor(0.)
        return torch.stack(conf_list)


class MeanProbExtractor_yolov8(nn.Module):
    def __init__(self, cls_id, num_cls, loss_target, conf_thres, iou_thres, max_det):
        super().__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = loss_target
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def forward(self, YOLOoutput):
        if isinstance(YOLOoutput, list):  # YOLOoutput: torch.Size([batch, 84, 8400])
            YOLOoutput = YOLOoutput[0]
        # YOLOoutput = nms_y8(YOLOoutput, conf_thres=self.conf_thres, iou_thres=self.iou_thres,
        #                     classes=self.cls_id, max_det=self.max_det, in_place=False)
        YOLOoutput = YOLOoutput.transpose(1, 2).contiguous()  # YOLOoutput: torch.Size([batch, 8400, 84]
        conf_list = []
        for box in YOLOoutput:
            if len(box) == 0:
                continue
            # conf_list.append(box[:, 4].mean())
            # i = torch.argmax(box[:, 4:], dim=1)
            # mask = i == self.cls_id
            # fbox = box[mask]
            # tgt_box = fbox[fbox[:, 4 + self.cls_id] > self.conf_thres]
            # if len(tgt_box) == 0:
            #     continue
            # conf_list.append(tgt_box[:, 4 + self.cls_id].mean())
        if len(conf_list) == 0:
            return torch.tensor(0.)
        return torch.stack(conf_list)


class MeanProbExtractor_frcnn(nn.Module):
    def __init__(self, cls_id, num_cls, loss_target, conf_thres, iou_thres, max_det):
        super().__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = loss_target
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def forward(self, output):
        conf_list = []
        for preds in output:
            boxes, labels, scores = preds['boxes'], preds['labels'], preds['scores']
            mask_label = labels == (self.cls_id + 1)
            filter_scores = scores[mask_label]
            if len(filter_scores) == 0:
                continue
            conf_list.append(filter_scores.mean())
        if len(conf_list) == 0:
            return torch.tensor(0.)
        return torch.stack(conf_list)


def preds2boxes(cfg, output):
    norm_preds = []
    if cfg.mode == 'yolov2':
        # [B,N,7(x,y,w,h,conf,cls_prob,cls)]
        boxes = get_region_boxes(output, cfg.conf_thres,
                                 cfg.model.num_classes,
                                 cfg.model.anchors,
                                 cfg.model.num_anchors)
        boxes = [nms(b, cfg.iou_thres) for b in boxes]
        for box in boxes:
            if len(box) == 0:
                norm_preds.append(torch.empty([0, 6]))
                continue
            box = torch.stack(box)
            box = box[box[:, -1] == cfg.cls_id].cuda()
            box[:, 5:6] *= box[:, 4:5]
            box = torch.cat([box[:, :4], box[:, 5:]], dim=1)
            norm_preds.append(box)
    elif cfg.mode == 'yolov5':
        if isinstance(output, tuple):
            output = output[0]
        bboxes = nms_y5(output, cfg.conf_thres, cfg.iou_thres, classes=cfg.cls_id, agnostic=False,
                        max_det=cfg.max_det)
        for boxes in bboxes:
            if len(boxes) > 0:
                boxes[:, :4] = xyxy2xywh(boxes[:, :4])
                boxes[:, :4] /= torch.tensor([*cfg.imgsz, *cfg.imgsz], device=cfg.device)
            norm_preds.append(boxes)
    elif cfg.mode == 'yolov8':
        if isinstance(output, tuple):
            output = output[0]
        bboxes = nms_y8(output, cfg.conf_thres, cfg.iou_thres, classes=cfg.cls_id, agnostic=False,
                        max_det=cfg.max_det)
        for boxes in bboxes:
            if len(boxes) > 0:
                boxes[:, :4] = xyxy2xywh(boxes[:, :4])
                boxes[:, :4] /= torch.tensor([*cfg.imgsz, *cfg.imgsz], device=cfg.device)
            norm_preds.append(boxes)
    else:
        raise "Unsupported mode!"
    # [tensor([[x, y, w, h, conf,cls], ...]), ...]
    return norm_preds


def boxes2labs(preds):
    labs = []
    max_n_labels = max([len(pred) for pred in preds])
    for pred in preds:
        pred_lab = torch.cat([pred[:, 5:], pred[:, :4]], dim=1)
        pad_size = max_n_labels - pred_lab.shape[0]
        if pad_size > 0:
            pred_lab = torch.cat([pred_lab, torch.ones(pad_size, 5).cuda()])
        else:
            pred_lab = pred_lab[:max_n_labels]
        labs.append(pred_lab)
    labs = torch.stack(labs)
    return labs


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),
                                               requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self, min_contrast=0.8, max_contrast=1.2, min_brightness=-0.1, max_brightness=0.1,
                 noise_factor=0.1,
                 minangle=-20, maxangle=20, scale=0.2, offsetx=0.2, offsety=0.2):
        super(PatchTransformer, self).__init__()
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.noise_factor = noise_factor
        self.minangle = minangle / 180 * math.pi
        self.maxangle = maxangle / 180 * math.pi
        self.scale = scale
        self.offsetx = offsetx
        self.offsety = offsety
        self.medianpooler = MedianPool2d(7, same=True)
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''

    def forward(self, adv_patch, lab_batch, img_size, do_blur=True, do_rotate=True, rand_loc=True,
                do_aug=True,
                by_rectangle=True):
        if lab_batch.dim() == 2:
            lab_batch = lab_batch.unsqueeze(0)
        adv_patch_size = adv_patch.size()[-1]
        if adv_patch_size > img_size:  # > img_size(416)
            adv_patch = adv_patch.unsqueeze(0)
            adv_patch = F.interpolate(adv_patch, size=img_size)
            adv_patch = adv_patch[0]
        if do_blur:
            adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms
        # Create random contrast tensor
        if do_aug:
            contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
            contrast = contrast.cuda()

            # Create random brightness tensor
            brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
            brightness = brightness.cuda()

            # Create random noise tensor
            noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

            # Apply contrast/brightness/noise, clamp
            adv_batch = adv_batch * contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
        msk_batch = torch.ones_like(adv_batch)

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(self.scale)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(self.scale)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        if rand_loc:
            targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
            targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
            off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-self.offsetx, self.offsetx))
            target_x = target_x + off_x
            off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-self.offsety, self.offsety))
            target_y = target_y + off_y

        # For person
        # target_y = target_y - 0.05
        #########################################################################################
        # Put the patch in the upper of the target
        # target_y = target_y - 0.07 * (target_size / current_patch_size / 3 * 5)
        #########################################################################################
        # Adjust the patch size on the object. The bigger the scale, the bigger the patch size.
        # target_size /= 2.0

        ############################################################################################
        # scale = target_size / current_patch_size * 4.0  # patch outside targets
        scale = target_size / current_patch_size  # patch on targets
        ############################################################################################
        # scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        if by_rectangle:
            theta[:, 1, 1] = theta[:, 1, 1] / 1.5
            theta[:, 1, 2] = theta[:, 1, 2] / 1.5

        b_sh = adv_batch.shape
        grid = F.affine_grid(theta, adv_batch.shape)

        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        # adv_batch_t = torch.clamp(adv_batch_t, 0, 0.999999)
        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        # img = adv_batch_t
        # img = img[0, 0, :, :, :].detach().cpu()
        # img = transforms.ToPILImage()(img)
        # img.save("adv_batch_t.jpg")
        # img.show()
        # exit()
        return (adv_batch_t * msk_batch_t).squeeze(0), adv_batch_t.squeeze(0), msk_batch_t.squeeze(0)
        # return adv_batch_t.squeeze(0), None, None


# class PatchTransformer(nn.Module):
#     """PatchTransformer: transforms batch of patches
#
#     Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
#     contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
#     batch of labels, and pads them to the dimension of an image.
#
#     """
#
#     def __init__(self):
#         super(PatchTransformer, self).__init__()
#         self.min_contrast = 0.8
#         self.max_contrast = 1.2
#         self.min_brightness = -0.1
#         self.max_brightness = 0.1
#         self.noise_factor = 0.10
#         self.minangle = -20 / 180 * math.pi
#         self.maxangle = 20 / 180 * math.pi
#         self.medianpooler = MedianPool2d(7, same=True)
#         '''
#         kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
#                                          [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
#                                          [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
#                                          [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
#                                          [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
#         self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
#         '''
#
#     def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True, do_blur=True, do_aug=True):
#         # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
#         if lab_batch.dim() == 2:
#             lab_batch = lab_batch.unsqueeze(0)
#         if do_blur:
#             adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
#         # Determine size of padding
#         pad = (img_size - adv_patch.size(-1)) / 2
#         # Make a batch of patches
#         adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
#         adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
#         batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
#
#         if do_aug:
#             # Contrast, brightness and noise transforms
#             # Create random contrast tensor
#             contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
#             contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#             contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
#             contrast = contrast.cuda()
#
#             # Create random brightness tensor
#             brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
#             brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#             brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
#             brightness = brightness.cuda()
#
#             # Create random noise tensor
#             noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
#
#             # Apply contrast/brightness/noise, clamp
#             adv_batch = adv_batch * contrast + brightness + noise
#
#         adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
#
#         # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
#         cls_ids = torch.narrow(lab_batch, 2, 0, 1)
#         cls_mask = cls_ids.expand(-1, -1, 3)
#         cls_mask = cls_mask.unsqueeze(-1)
#         cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
#         cls_mask = cls_mask.unsqueeze(-1)
#         cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
#         # msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask
#         msk_batch = torch.where(cls_mask == -1, torch.tensor(0, dtype=torch.float), torch.tensor(1, dtype=torch.float))
#
#         # Pad patch and mask to image dimensions
#         mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
#         adv_batch = mypad(adv_batch)
#         msk_batch = mypad(msk_batch)
#
#         # Rotation and rescaling transforms
#         anglesize = (lab_batch.size(0) * lab_batch.size(1))
#         if do_rotate:
#             angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
#         else:
#             angle = torch.cuda.FloatTensor(anglesize).fill_(0)
#
#         # Resizes and rotates
#         current_patch_size = adv_patch.size(-1)
#         lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
#         lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
#         lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
#         lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
#         lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
#         target_size = torch.sqrt(
#             ((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
#         target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
#         target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
#         targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
#         targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
#         if (rand_loc):
#             off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.2, 0.2))
#             target_x = target_x + off_x
#             off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.2, 0.2))
#             target_y = target_y + off_y
#         #########################################################################################
#         # Put the patch in the upper of the target
#         # target_y = target_y - 0.07 * (target_size / current_patch_size / 3 * 5)
#         #########################################################################################
#         # Adjust the patch size on the object. The bigger the scale, the bigger the patch size.
#         target_size /= 2.0
#
#         ############################################################################################
#         # scale = target_size / current_patch_size * 4.0  # patch outside targets
#         scale = target_size / current_patch_size  # patch on targets
#         ############################################################################################
#
#         scale = scale.view(anglesize)
#
#         s = adv_batch.size()
#         adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
#         msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])
#
#         tx = (-target_x + 0.5) * 2
#         ty = (-target_y + 0.5) * 2
#         sin = torch.sin(angle)
#         cos = torch.cos(angle)
#
#         # Theta = rotation,rescale matrix
#         theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
#         theta[:, 0, 0] = cos / scale
#         theta[:, 0, 1] = sin / scale
#         theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
#         theta[:, 1, 0] = -sin / scale
#         theta[:, 1, 1] = cos / scale
#         theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
#
#         b_sh = adv_batch.shape
#         grid = F.affine_grid(theta, adv_batch.shape)
#
#         adv_batch_t = F.grid_sample(adv_batch, grid)
#         msk_batch_t = F.grid_sample(msk_batch, grid)
#
#         '''
#         # Theta2 = translation matrix
#         theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
#         theta2[:, 0, 0] = 1
#         theta2[:, 0, 1] = 0
#         theta2[:, 0, 2] = (-target_x + 0.5) * 2
#         theta2[:, 1, 0] = 0
#         theta2[:, 1, 1] = 1
#         theta2[:, 1, 2] = (-target_y + 0.5) * 2
#
#         grid2 = F.affine_grid(theta2, adv_batch.shape)
#         adv_batch_t = F.grid_sample(adv_batch_t, grid2)
#         msk_batch_t = F.grid_sample(msk_batch_t, grid2)
#
#         '''
#         adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
#         msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])
#
#         adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
#         # img = adv_batch_t * msk_batch_t
#         # img = img[0, 0, :, :, :].detach().cpu()
#         # img = transforms.ToPILImage()(img)
#         # img.save("adv_batch_t.jpg")
#         # img.show()
#         # exit()
#
#         return adv_batch_t * msk_batch_t

class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    # def forward(self, img_batch, adv_batch):
    #     advs = torch.unbind(adv_batch, 1)
    #     for adv in advs:
    #         img_batch = torch.where((adv == 0), img_batch, adv)
    #     return img_batch
    def forward(self, img_batch, adv_batch, lab_idx):
        for i in range(len(img_batch)):
            advs = adv_batch[lab_idx == i]
            for adv in advs:
                img_batch[i] = torch.where((adv == 0), img_batch[i], adv)
        return img_batch


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.

    """

    def __init__(self, img_dir, lab_dir, imgsize, cls_ids=0, pad=False):
        if isinstance(cls_ids, int):
            cls_ids = [cls_ids]
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_bmp_images = len(fnmatch.filter(os.listdir(img_dir), '*.bmp'))
        n_images = n_png_images + n_jpg_images + n_bmp_images
        self.pad = pad
        self.len = n_images
        self.img_dir = img_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir),
                                                                                       '*.jpg') + fnmatch.filter(
            os.listdir(img_dir), '*.bmp')
        self.img_paths = []
        self.lab_paths = []
        self.lab_dir = lab_dir
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        if lab_dir is None:
            return
        img_names = []
        label_cache = {}
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        # self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt').replace(
                '.bmp', '.txt')
            labels = np.loadtxt(lab_path, ndmin=2)
            if cls_ids is None:
                self.lab_paths.append(lab_path)
                img_names.append(img_name)
            else:
                clss = labels[:, 0]
                idxes = []
                for i, cls in enumerate(clss):
                    if cls in cls_ids:
                        idxes.append(i)
                if len(idxes) > 0:
                    label_cache[img_name] = idxes
                    img_names.append(img_name)
                    self.lab_paths.append(lab_path)
        self.img_names = img_names
        self.label_cache = label_cache
        print(f'Total images count: {len(self.img_names)}')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        if self.lab_dir is None:
            image, _ = self.pad_and_scale(image)
            return to_tensor(image), None, None, (img_path, None)
        lab_path = (os.path.join(self.lab_dir, self.img_names[idx])
                    .replace('.jpg', '.txt')
                    .replace('.png', '.txt')
                    .replace('.bmp', '.txt'))
        label = np.loadtxt(lab_path, ndmin=2)
        if len(self.label_cache) != 0:
            label = label[self.label_cache[self.img_names[idx]]]
        label = torch.from_numpy(label).float()
        label_idx = torch.zeros(label.shape[0])
        image, label = self.pad_and_scale(image, label)
        image = to_tensor(image)
        return image, label, label_idx, (img_path, lab_path)

    def pad_and_scale(self, img, lab=None):
        resize = transforms.Resize((self.imgsize, self.imgsize))
        if not self.pad:
            img = resize(img)  # choose here
            return img, lab
        w, h = img.size
        if w != h:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                img = padded_img
                if lab is not None:
                    lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                    lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                img = padded_img
                if lab is not None:
                    lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                    lab[:, [4]] = (lab[:, [4]] * h / w)
        img = resize(img)  # choose here
        return img, lab

    # def pad_lab(self, lab):
    #     pad_size = self.max_n_labels - lab.shape[0]
    #     if (pad_size > 0):
    #         # padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
    #         padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=-1)
    #     else:
    #         padded_lab = lab
    #     return padded_lab


def collate_fn(batch):
    """Batches images, labels, paths, and shapes, assigning unique indices to targets in merged label tensor."""
    b = zip(*batch)
    im, label, label_idx, path = zip(*batch)  # transposed
    path = np.array(path).transpose()
    for i, lb in enumerate(label):
        if lb is None:
            return torch.stack(im, 0), None, None, path
        label_idx[i][:] = i  # add target image index for build_targets()
    return torch.stack(im, 0), torch.cat(label, 0), torch.cat(label_idx, 0), path


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    torch.manual_seed(0)

    cls_ids = 3
    dataset = InriaDataset('dataset/sandtable_pylon/val/images',
                           'dataset/sandtable_pylon/val/labels_yolov5s_st',
                           640,
                           cls_ids=cls_ids)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    patch = torch.randn((3, 512, 512), dtype=torch.float).cuda()
    transformer = PatchTransformer().cuda()
    applier = PatchApplier().cuda()
    for i, (imgs, labs, idxes, paths) in enumerate(loader):
        imgs = imgs.cuda()
        labs = labs.cuda()
        idxes = idxes.cuda()
        adv_batch = transformer(patch, labs, 640,
                                rand_loc=False,
                                do_aug=False,
                                do_rotate=False,
                                do_blur=False,
                                by_rectangle=False)
        p_img_batch = applier(imgs, adv_batch, idxes)
        r = torchvision.utils.make_grid(p_img_batch, nrow=int(math.sqrt(p_img_batch.shape[0])), padding=0)
        torchvision.utils.save_image(r, 'test.jpg')
        break
