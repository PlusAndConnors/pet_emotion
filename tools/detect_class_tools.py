# Copyright (c) OpenMMLab. All rights reserved.
# this is for crop_dog_face using cpu
import argparse
import ast
import io
import os
import os.path as osp
from glob import glob
from typing import List

import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
# try:
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
from torchvision.transforms import transforms

from detect import models

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

EYE_FACE_RATIO = 61.44


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


class DogFaceDetect():
    def __init__(self, args=None):
        # det
        if not args.crop_face:
            self.detect_model = init_detector(args.det_config, args.det_checkpoint, args.device)
            self.det_label = 15 if args.dog_cat == 'cat' else 16
            assert self.detect_model.CLASSES[self.det_label] == args.dog_cat, (
                'We require you to use a detector ''trained on COCO')
            self.pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
        # rec
        self.model_dict, self.model_name, self.models = load_cls_model(args)
        self.model_dict, self.model_name, self.models = load_cls_model(args)
        self.lengh, self.labels = len(self.model_dict), list()

        self.args = args
        self.img_size = args.output_size
        self._transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), ])
        print('fininsh_model_load')

    def one_img2cropfaces(self, img_path, label=None):
        t = time.time()
        det_result = inference_detector(self.detect_model, img_path)[self.det_label]
        print(time.time() - t)
        det_check = det_result[:, 4] >= self.args.det_score_thr
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR) if isinstance(img_path, str) else img_path
        crop_face_imgs = list()
        if any(det_check):
            det_result = det_result[det_check]
            det_result = [dict(bbox=x) for x in list(det_result)]
            t = time.time()
            pose = inference_top_down_pose_model(self.pose_model, img_path, det_result, format='xyxy')[0]
            print(time.time() - t)
            for pos in pose:
                img = facecrop(pos, img_raw, (self.img_size, self.img_size))
                crop_face_imgs.append(img)
        else:
            crop_face_imgs, img_raw, det_check, pose = [], [], [], []
        if self.args.test == 'test':
            img_raw = vis_pose_result(self.pose_model, img_raw, pose)
        return crop_face_imgs, img_raw, det_check, pose

    def cropface2feature(self, images):
        one_feature, prediction_list = dict(), list()
        images = cv2.resize(images, (224, 224))
        images = self._transform(images)
        images = images.to(self.args.device)
        images = torch.stack([images], 0)

        for i, model in enumerate(self.models):
            with torch.no_grad():
                outputs = model(images).cpu()
                outputs = F.softmax(outputs, 1)
                outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

                outputs = [round(o, 4) for o in outputs.numpy()]
                prediction_list.append(outputs)
                one_feature[self.model_name[i]] = outputs
        return one_feature, np.array(prediction_list)

    def feature2result(self, feature):
        model_dict, args, model_dict_proba = self.model_dict, self.args, self.args.proba_conf
        test_results_list, tmp_test_result_list = list(), list()

        for idx, (model_name, _) in enumerate(model_dict):
            tmp_test_result_list.append(model_dict_proba[idx] * np.array(feature[idx]))
        tmp_test_result_list = np.array(tmp_test_result_list)
        y_score = np.sum(tmp_test_result_list, axis=0)
        y_pred = np.argmax(y_score, axis=0)

        return y_pred, max(y_score / sum(y_score))

    def show_each_model_result(self, feature):
        model_dict, args, model_dict_proba = self.model_dict, self.args, self.args.proba_conf
        test_results_list, tmp_test_result_list, y_pred = list(), list(), dict()
        for idx, (model_name, _) in enumerate(model_dict):
            y_score = feature[idx]
            y_pred[model_name] = (np.argmax(y_score, axis=0), max(y_score))
        print(y_pred)
        return y_pred

    def one_img2bbox_check(self, img):
        det_result = inference_detector(self.detect_model, img)
        det_check = [(i, v) for i, v in enumerate(det_result) if len(v) > 0]
        final_det = list()
        for i, vv in det_check:
            det_check_b = vv[:, 4] > self.args.det_score_thr
            if any(det_check_b):
                final_det.append((i, vv[det_check_b]))
        for classes, bboxes in final_det:
            for box in bboxes:
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img, str(classes), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return img

    def init_set(self, args):
        self.args = args
        self.img_size = args.output_size
        # det
        if not args.crop_face:
            self.detect_model = init_detector(args.det_config, args.det_checkpoint, args.device)
            self.det_label = 15 if args.dog_cat == 'cat' else 16
            assert self.detect_model.CLASSES[self.det_label] == args.dog_cat, (
                'We require you to use a detector ''trained on COCO')
            self.pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
        # rec
        checkpoints = glob(os.path.join(args.cls_check_path, '*'))
        self.model_name = []
        self.model_dict = [(i.split('/')[-1].split('__')[0], i.split('/')[-1]) for i in
                           checkpoints if not i.split('/')[-1].split('__')[0] == i.split('/')[-1]]
        # load_label + img_data
        labels, model_set = list(), list()
        self.lengh = len(self.model_dict)
        self.labels = labels
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        for model_name, checkpoint_path in self.model_dict:
            # each item is 7-ele array
            print("Processing", checkpoint_path)

            model = getattr(models, model_name)
            self.model_name.append(model_name)
            model = model(in_channels=3, num_classes=4)
            state = torch.load(os.path.join(self.args.cls_check_path, checkpoint_path),
                               map_location=lambda storage, loc: storage)
            model.load_state_dict(state["net"])
            model.to(self.args.device)
            model.eval()
            model_set.append(model)
        self.models = model_set


def load_det_model(args):
    detect_model = init_detector(args.det_config, args.det_checkpoint, args.device)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    return detect_model, pose_model


def load_cls_model(args):
    checkpoints = glob(os.path.join(args.cls_check_path, '*'))
    model_names, model_set = list(), list()
    model_dict = [(i.split('/')[-1].split('__')[0], i.split('/')[-1]) for i in checkpoints if
                  not i.split('/')[-1].split('__')[0] == i.split('/')[-1]]
    # load_label + img_data
    labels, model_set = list(), list()

    for model_name, checkpoint_path in model_dict:
        # each item is 7-ele array
        print("Processing", checkpoint_path)

        model = getattr(models, model_name)
        model_names.append(model_name)
        model = model(in_channels=3, num_classes=4)
        state = torch.load(os.path.join(args.cls_check_path, checkpoint_path),
                           map_location=lambda storage, loc: storage)
        model.load_state_dict(state["net"])
        model.to(args.device)
        model.eval()
        model_set.append(model)

    return model_dict, model_names, model_set


def facecrop(p, img_raw, img_size):
    (rx, ry), (lx, ly) = p['keypoints'][0:2, 0:2]
    center = ((lx + rx) // 2, (ly + ry) // 2)
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    scale = EYE_FACE_RATIO / (np.sqrt(((rx - lx) ** 2) + ((ry - ly) ** 2)))
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += (img_size[0] / 2 - center[0])
    M[1, 2] += (img_size[1] / 2 + int(img_size[1] / 20) - center[1])
    img = cv2.warpAffine(img_raw, M, img_size, borderValue=0.0)
    return img


def show_result_img(img):
    img = cv2.imencode('.png', img)[1].tostring()
    f = io.BytesIO()
    f.write(img)
    f.seek(0)
    return f


def make_byte_image_2_cv2(image_file):
    image_bytes = image_file.read()
    image_cv = np.fromstring(image_bytes, np.uint8)
    img = cv2.imdecode(image_cv, cv2.IMREAD_COLOR)
    return img


def draw_result(img, pose, label):
    diction = ['중립/안정', '행복/놀람', '슬픔/두려움', '화남/싫음']

    bbox = pose['bbox']
    boxx, boxy = int(bbox[0]), int(bbox[1])
    # land = pose['keypoints']
    predict = diction[label]
    # cv2.rectangle(img, (boxx, boxy), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
    img = cv2_draw_korea(img, str(predict), (boxx, boxy + 12))
    # for key in land:
    # cv2.circle(img, (boxx + int(key[0]), boxy + int(key[1])), 1, (0, 255, 255), 4)
    return img


def cv2_draw_korea(cv_img, text, position=(10, 0)):
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    b, g, r, a = 255, 255, 255, 0
    org = position
    font = ImageFont.truetype(font="./tools/gongso.ttf", size=int(sum(org) / 20))  # korea font
    im_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(im_pil)
    draw.text(org, text, (r, g, b, a), font=font)  # because of RGB2BGR
    img = np.array(im_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def for_same_name(path: str, name: str, data_type='jpg') -> str:
    save_path, uni = osp.join(f'{path}', f'{name}.{data_type}'), 1
    while osp.exists(save_path):
        save_path = osp.join(f'{path}', f'{name}_({uni}).{data_type}')
        uni += 1
    return save_path


def who_r_u(score: List[List[int]], result: List[List[dict]], threshold=38.5) -> List[List[dict]]:
    for z, score_cut in enumerate(score):
        for y, scc in enumerate(score_cut):
            if scc < threshold:
                result[z][y]['label'] = 'who are you?'
    return result

# def main():
#     from glob import glob
#     api_test = DogFaceDetect()
#     data = glob('../../test-dog/cropdata/final6/testset/*')
#     result, predicts, confs = dict(), list(), list()
#
#     for img_path in data:
#         t = time.time()
#         cropface = api_test.one_img2cropface(img_path)
#         if len(cropface[0]):
#             features, one_feature = api_test.cropface2feature(cropface)
#             y_pred, conf = api_test.feature2result(features)
#             predicts.append(y_pred), confs.append(conf)
#             print(time.time() - t)
#     result["y_pred"], result["confidence"] = predicts, confs
#     print('f')
#
#
# if __name__ == '__main__':
#     main()
