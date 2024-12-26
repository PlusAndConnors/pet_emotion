import json
import sys
import time

import cv2
import os.path as osp
import numpy as np
from flask import request, send_file

from .detect_class_tools import DogFaceDetect, make_byte_image_2_cv2, show_result_img, draw_result, load_det_model


def set_class(args, class_name):
    args.test = 'test' if 'test' in class_name else 0
    args.test = 'model_test' if 'model_test' in class_name else args.test
    args.crop_face = True if 'crop' in class_name else False
    args.dog_cat = 'cat' if '고양이' == class_name else 'dog'
    args.test = 'not_animal' if 'not_animal' in class_name else args.test
    return args


def emotion_predict(args, models):
    if not request.method == "POST":
        sys.exit()

    if request.files.getlist("image"):  # and len(request.files.getlist("image")) > 0:  # multi image
        image_files = request.files.getlist('image')
        if request.values.get('classname'):
            class_name = request.values.get('classname')
            args = set_class(args, class_name)
        result, predicts, confs, features, pose = dict(), list(), list(), list(), list()

        for i, file in enumerate(image_files):
            name = file.filename.split('.')[0]
            t = time.time()
            img = make_byte_image_2_cv2(file)

            if args.test == 'not_animal':
                result = models.one_img2bbox_check(img)
                cv2.imwrite(osp.join('check_what', f'{name}_{i}.jpg'), result)
                continue
            if not args.crop_face:
                cropfaces, img, box_chk, pose = models.one_img2cropfaces(img)
            else:
                cropfaces = [img]

            for i, cropface in enumerate(cropfaces):
                feature, one_feature = models.cropface2feature(cropface)
                y_pred, conf = models.feature2result(one_feature)
                if args.test == 'model_test':
                    check = models.show_each_model_result(one_feature)
                predicts.append(y_pred), confs.append(round(conf, 3))
                args.test == 'test' and len(pose)
                if args.test == 'test' and len(pose):
                    # img = draw_result(img, pose[i], y_pred)
                    img_file = show_result_img(img)
                if not args.save_img_path == '':
                    save_result = cropface if args.test == 'test' else img
                    cv2.imwrite(osp.join(args.save_img_path, f'{name}_{i}.jpg'), save_result)
            print('processing times = ', time.time() - t)
        if args.test == 'not_animal':
            img_file = show_result_img(img)
            return send_file(img_file, mimetype='image/png')
        result["y_pred"], result["confidence"] = predicts, confs
        print(result)
        return json.dumps(result, cls=NpEncoder) if not args.test == 'test' else send_file(img_file,
                                                                                           mimetype='image/png')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
