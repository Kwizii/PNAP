import numpy as np
import torch
import torch.nn as nn

from post_util import get_region_boxes
from utils.general import non_max_suppression_gradable


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

    def forward(self, YOLOoutput):  # YOLOoutput: torch.Size([batch, 64512, 20])
        if isinstance(YOLOoutput, list):
            YOLOoutput = YOLOoutput[0]
        output = YOLOoutput.transpose(1, 2).contiguous()  # [batch, 20, 64512]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 64512]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 15, 64512]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (Plane)
        confs_for_class = normal_confs[:, self.cls_id, :]
        # confs_if_object = output_objectness  # confs_for_class * output_objectness
        # confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.loss_target(output_objectness, confs_for_class)
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
