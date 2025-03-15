import imghdr  # get_image_size
import math
import os
import struct  # get_image_size
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import torchvision.transforms as T

from utils.general import xywh2xyxy
from utils.metrics import box_iou

if os.name == 'nt':  # Windows
    font_path = "C:\\Windows\\Fonts\\Arial.ttf"
else:  # Unix/Linux/Mac
    font_path = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"

def sigmoid(x):
    return 1.0 / (math.exp(-x) + 1.)


def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x / x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    # print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=False, validation=False):
    # anchor_step = len(anchors)/num_anchors
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []
    # print(output.size())
    output = output.view(batch * num_anchors, 5 + num_classes, h * w)
    # print(output.size())
    output = output.transpose(0, 1).contiguous()
    # print(output.size())
    output = output.view(5 + num_classes, batch * num_anchors * h * w)
    # print(output.size())
    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).to(output.device)
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).to(output.device)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).to(output.device)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).to(output.device)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1))).data
    # print(cls_confs.size())
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(torch.stack(box))
        if len(boxes) == 0:
            all_boxes.append(torch.zeros([0, 7]))
        else:
            all_boxes.append(torch.stack(boxes))
    t3 = time.time()
    # if False:
    #     print('---------------------------------')
    #     print('matrix computation : %f' % (t1 - t0))
    #     print('        gpu to cpu : %f' % (t2 - t1))
    #     print('      boxes filter : %f' % (t3 - t2))
    #     print('---------------------------------')
    return all_boxes


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2] / 2.0) * width))
        y1 = int(round((box[1] - box[3] / 2.0) * height))
        x2 = int(round((box[0] + box[2] / 2.0) * width))
        y2 = int(round((box[1] + box[3] / 2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def limit_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return x1, y1, x2, y2


def plot_boxes(img, boxes, savename=None, class_names=None, savedir=None, fill=False):
    colors = torch.FloatTensor([[0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]])
    boxes = boxes.detach().cpu().numpy()
    if boxes.shape[-1] == 5:
        boxes = boxes[:, [1, 2, 3, 4, 0]]
        # boxes[:, :4] = xywh2xyxy(boxes[:, :4])

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    if isinstance(img, torch.Tensor):
        img = T.ToPILImage()(img.detach())
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img, mode='RGBA')
    for i in range(len(boxes)):
        # x,y,w,h,obj_conf,cls_conf,cls
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height
        x1, y1, x2, y2 = limit_box((x1, y1, x2, y2), width, height)
        rgb = (255, 0, 0)
        rect_width = 3
        box_width = box[2] * width
        ttf = ImageFont.truetype(font_path, max(min(int(box_width * 0.2), 20), 6))
        if len(box) >= 5 and class_names:
            if len(box) == 6:
                conf = float(box[4])
                cls_id = int(box[5])
                text = f"{class_names[cls_id]} {conf:.2f}"
            else:
                conf = 1
                cls_id = int(box[4])
                text = f"{class_names[cls_id]}"
            ###
            classes = len(class_names)
            rect_fill = None
            if fill:
                # 根据百分比计算当前颜色
                if conf <= 0.45:
                    red = 255
                    green = 0  # 红色到黄色的渐变
                elif conf >= 0.85:
                    red = int(255 * ((0.85 - conf) / (0.85 - 0.45)))  # 绿色到黄色的渐变
                    green = 255
                else:
                    red = 255
                    green = 0
                rect_fill = (red, green, 0, 156)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                rgb = (red, green, blue)
            # print('[%i]%s: %f' % (cls_id, class_names[cls_id], conf))
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=rect_width, fill=rect_fill)
            draw.text((x1 + 2 + rect_width, y1 + 2 + rect_width), text, font=ttf, fill=rgb)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=rect_width)
    if savename:
        print("save plot results to %s" % savename)
        img.save(os.path.join(savedir, savename))
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size // 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    # remove truths of which the width is smaller then the min_box_scale
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    # torch.Size([1, 100, 32, 32])
    output = model.forward(img)  # Simen: dit doet een forward, vervangen voor duidelijkheid
    # output = output.data
    # for j in range(100):
    #    sys.stdout.write('%f ' % (output.storage()[j]))
    # print('')
    t3 = time.time()

    # debug
    print(output.view)

    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)[0]
    # for j in range(len(boxes)):
    # print(boxes[j])
    t4 = time.time()

    boxes = nms(boxes, nms_thresh)
    t5 = time.time()

    # if False:
    #     print('-----------------------------------')
    #     print(' image to tensor : %f' % (t1 - t0))
    #     print('  tensor to cuda : %f' % (t2 - t1))
    #     print('         predict : %f' % (t3 - t2))
    #     print('get_region_boxes : %f' % (t4 - t3))
    #     print('             nms : %f' % (t5 - t4))
    #     print('           total : %f' % (t5 - t0))
    #     print('-----------------------------------')
    return boxes


def do_detect_yolov5(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    # torch.Size([1, 64512, 20])
    output = model.forward(img)  # Simen: dit doet een forward, vervangen voor duidelijkheid
    # output = output.data
    # for j in range(100):
    #    sys.stdout.write('%f ' % (output.storage()[j]))
    # print('')
    t3 = time.time()

    # debug
    print(output.view)

    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)[0]
    # for j in range(len(boxes)):
    # print(boxes[j])
    t4 = time.time()

    boxes = nms(boxes, nms_thresh)
    t5 = time.time()

    # if False:
    #     print('-----------------------------------')
    #     print(' image to tensor : %f' % (t1 - t0))
    #     print('  tensor to cuda : %f' % (t2 - t1))
    #     print('         predict : %f' % (t3 - t2))
    #     print('get_region_boxes : %f' % (t4 - t3))
    #     print('             nms : %f' % (t5 - t4))
    #     print('           total : %f' % (t5 - t0))
    #     print('-----------------------------------')
    return boxes


def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets


def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192 * 1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close()
    return count


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                    # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height


def grid_images(images, max_num=9, max_len=1280):
    n = len(images)
    if max_num is not None:
        n = min(max_num, n)
    rows = int(math.sqrt(n))
    while n % rows != 0:
        rows -= 1
    cols = n // rows
    while rows * cols < n:
        cols += 1
    width, height = images[0].size
    grid = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        grid.paste(img, (width * (i % cols), height * (i // cols)))
    if max_len is not None:
        if width > height:
            new_width = max_len
            new_height = int(height * new_width / width)
        else:
            new_height = max_len
            new_width = int(width * new_height / height)
        grid = grid.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return grid


def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
