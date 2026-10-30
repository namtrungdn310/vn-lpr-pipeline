# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tiền/hậu xử lý PaddleX gốc; bỏ decorator, docstring và nhánh crop đa giác."""

import logging
import math
import re
from typing import List, Optional, Union
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

class DetResizeForTest:

    def __init__(self, input_shape=None, max_side_limit=4000, **kwargs):
        self.resize_type = 0
        self.keep_ratio = False
        if input_shape is not None:
            self.input_shape = input_shape
            self.resize_type = 3
        elif 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
            if 'keep_ratio' in kwargs:
                self.keep_ratio = kwargs['keep_ratio']
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'
        self.max_side_limit = max_side_limit

    def __call__(self, imgs, limit_side_len: Union[int, None]=None, limit_type: Union[str, None]=None, max_side_limit: Union[int, None]=None):
        max_side_limit = max_side_limit if max_side_limit is not None else self.max_side_limit
        resize_imgs, img_shapes = ([], [])
        for ori_img in imgs:
            img, shape = self.resize(ori_img, limit_side_len, limit_type, max_side_limit)
            resize_imgs.append(img)
            img_shapes.append(shape)
        return (resize_imgs, img_shapes)

    def resize(self, img, limit_side_len: Union[int, None], limit_type: Union[str, None], max_side_limit: Union[int, None]=None):
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)
        if self.resize_type == 0:
            img, [ratio_h, ratio_w] = self.resize_image_type0(img, limit_side_len, limit_type, max_side_limit)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        elif self.resize_type == 3:
            img, [ratio_h, ratio_w] = self.resize_image_type3(img)
        else:
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        return (img, np.array([src_h, src_w, ratio_h, ratio_w]))

    def image_padding(self, im, value=0):
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]
        if self.keep_ratio is True:
            resize_w = ori_w * resize_h / ori_h
            N = math.ceil(resize_w / 32)
            resize_w = N * 32
        if resize_h == ori_h and resize_w == ori_w:
            return (img, [1.0, 1.0])
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return (img, [ratio_h, ratio_w])

    def resize_image_type0(self, img, limit_side_len: Union[int, None], limit_type: Union[str, None], max_side_limit: Union[int, None]=None):
        limit_side_len = limit_side_len or self.limit_side_len
        limit_type = limit_type or self.limit_type
        h, w, c = img.shape
        if limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif limit_type == 'min':
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif limit_type == 'resize_long':
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception('not support limit type, image ')
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        if max(resize_h, resize_w) > max_side_limit:
            logging.warning(f'Resized image size ({resize_w}x{resize_h}) exceeds max_side_limit of {max_side_limit}. Resizing to fit within limit.')
            ratio = float(max_side_limit) / max(resize_h, resize_w)
            resize_h, resize_w = (int(resize_h * ratio), int(resize_w * ratio))
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        if resize_h == h and resize_w == w:
            return (img, [1.0, 1.0])
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return (None, (None, None))
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            logging.info(img.shape, resize_w, resize_h)
            raise
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return (img, [ratio_h, ratio_w])

    def resize_image_type2(self, img):
        h, w, _ = img.shape
        resize_w = w
        resize_h = h
        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        if resize_h == h and resize_w == w:
            return (img, [1.0, 1.0])
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return (img, [ratio_h, ratio_w])

    def resize_image_type3(self, img):
        resize_c, resize_h, resize_w = self.input_shape
        ori_h, ori_w = img.shape[:2]
        if resize_h == ori_h and resize_w == ori_w:
            return (img, [1.0, 1.0])
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return (img, [ratio_h, ratio_w])

class NormalizeImage:

    def __init__(self, scale=None, mean=None, std=None, order='chw'):
        super().__init__()
        if isinstance(scale, str):
            scale = eval(scale)
        self.order = order
        scale = scale if scale is not None else 1.0 / 255.0
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]
        self.alpha = [scale / std[i] for i in range(len(std))]
        self.beta = [-mean[i] / std[i] for i in range(len(std))]

    def __call__(self, imgs):

        def _norm(img):
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            if self.order == 'chw':
                img = np.transpose(img, (2, 0, 1))
            split_im = list(cv2.split(img))
            for c in range(img.shape[2]):
                split_im[c] = split_im[c].astype(np.float32)
                split_im[c] *= self.alpha[c]
                split_im[c] += self.beta[c]
            res = cv2.merge(split_im)
            if self.order == 'chw':
                res = np.transpose(res, (1, 2, 0))
            return res
        return [_norm(img) for img in imgs]

class DBPostProcess:

    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=2.0, use_dilation=False, score_mode='fast', box_type='quad', **kwargs):
        super().__init__()
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in ['slow', 'fast'], 'Score mode must be in [slow, fast] but got: {}'.format(score_mode)
        self.use_dilation = use_dilation

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height, box_thresh, unclip_ratio):
        bitmap = _bitmap
        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height
        boxes = []
        scores = []
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if box_thresh > score:
                continue
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            if len(box) > 0:
                _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
                if sside < self.min_size + 2:
                    continue
            else:
                continue
            box = np.array(box)
            for i in range(box.shape[0]):
                box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
                box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))
            boxes.append(box)
            scores.append(score)
        return (boxes, scores)

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height, box_thresh, unclip_ratio):
        bitmap = _bitmap
        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = (outs[0], outs[1], outs[2])
        elif len(outs) == 2:
            contours, _ = (outs[0], outs[1])
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == 'fast':
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if box_thresh > score:
                continue
            box = self.unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            for i in range(box.shape[0]):
                box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
                box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return (np.array(boxes, dtype=np.int16), scores)

    def unclip(self, box, unclip_ratio):
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, True)
        distance = area * unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        try:
            expanded = np.array(offset.Execute(distance))
        except ValueError:
            expanded = np.array(offset.Execute(distance)[0])
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = (0, 1, 2, 3)
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return (box, min(bounding_box[1]))

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
        xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
        ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
        ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))
        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, preds, img_shapes, thresh: Union[float, None]=None, box_thresh: Union[float, None]=None, unclip_ratio: Union[float, None]=None):
        boxes, scores = ([], [])
        for pred, img_shape in zip(preds[0], img_shapes):
            box, score = self.process(pred, img_shape, thresh or self.thresh, box_thresh or self.box_thresh, unclip_ratio or self.unclip_ratio)
            boxes.append(box)
            scores.append(score)
        return (boxes, scores)

    def process(self, pred, img_shape, thresh, box_thresh, unclip_ratio):
        pred = pred[0, :, :]
        segmentation = pred > thresh
        dilation_kernel = None if not self.use_dilation else np.array([[1, 1], [1, 1]])
        src_h, src_w, ratio_h, ratio_w = img_shape
        if dilation_kernel is not None:
            mask = cv2.dilate(np.array(segmentation).astype(np.uint8), dilation_kernel)
        else:
            mask = segmentation
        if self.box_type == 'poly':
            boxes, scores = self.polygons_from_bitmap(pred, mask, src_w, src_h, box_thresh, unclip_ratio)
        elif self.box_type == 'quad':
            boxes, scores = self.boxes_from_bitmap(pred, mask, src_w, src_h, box_thresh, unclip_ratio)
        else:
            raise ValueError("box_type can only be one of ['quad', 'poly']")
        return (boxes, scores)

def validate_text_rec_image_array(img: np.ndarray, index: Optional[int]=None) -> None:
    if img.ndim < 2:
        raise ValueError(f'Text recognition expects images with at least 2 dimensions; got shape {tuple(img.shape)}.')
    h, w = (int(img.shape[0]), int(img.shape[1]))
    if h <= 0 or w <= 0 or img.size == 0:
        suffix = f' at input index {index}' if index is not None else ''
        raise ValueError(f'Text recognition requires images with positive height and width; got shape {tuple(img.shape)}{suffix}.')

class OCRReisizeNormImg:

    def __init__(self, rec_image_shape=[3, 48, 320], input_shape=None):
        super().__init__()
        self.rec_image_shape = rec_image_shape
        self.input_shape = input_shape
        self.max_imgW = 3200

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int(imgH * max_wh_ratio)
        if imgW > self.max_imgW:
            resized_image = cv2.resize(img, (self.max_imgW, imgH))
            resized_w = self.max_imgW
            imgW = self.max_imgW
        else:
            h, w = img.shape[:2]
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, imgs):
        for i, img in enumerate(imgs):
            validate_text_rec_image_array(img, index=i)
        if self.input_shape is None:
            return [self.resize(img) for img in imgs]
        else:
            return [self.staticResize(img) for img in imgs]

    def resize(self, img):
        imgC, imgH, imgW = self.rec_image_shape
        max_wh_ratio = imgW / imgH
        h, w = img.shape[:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        img = self.resize_norm_img(img, max_wh_ratio)
        return img

    def staticResize(self, img):
        imgC, imgH, imgW = self.input_shape
        resized_image = cv2.resize(img, (int(imgW), int(imgH)))
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

class BaseRecLabelDecode:

    def __init__(self, character_str=None, use_space_char=True):
        super().__init__()
        self.reverse = False
        character_list = list(character_str) if character_str is not None else list('0123456789abcdefghijklmnopqrstuvwxyz')
        if use_space_char:
            character_list.append(' ')
        character_list = self.add_special_char(character_list)
        self.dict = {}
        for i, char in enumerate(character_list):
            self.dict[char] = i
        self.character = character_list

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)
        return ''.join(pred_re[::-1])

    def add_special_char(self, character_list):
        return character_list

    def get_word_info(self, text, selection):
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection == True)[0]
        for c_i, char in enumerate(text):
            if '一' <= char <= '鿿':
                c_state = 'cn'
            elif bool(re.search('[a-zA-Z0-9]', char)):
                c_state = 'en&num'
            else:
                c_state = 'symbol'
            if char == '.' and state == 'en&num' and (c_i + 1 < len(text)) and bool(re.search('[0-9]', text[c_i + 1])):
                c_state = 'en&num'
            if char == '-' and state == 'en&num':
                c_state = 'en&num'
            if state is None:
                state = c_state
            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state
            word_content.append(char)
            word_col_content.append(int(valid_col[c_i]))
        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)
        return (word_list, word_col_list, state_list)

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False, return_word_box=False):
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            char_list = [self.character[text_id] for text_id in text_index[batch_idx][selection]]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
            text = ''.join(char_list)
            if self.reverse:
                text = self.pred_reverse(text)
            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(text, selection)
                result_list.append((text, np.mean(conf_list).tolist(), [len(text_index[batch_idx]), word_list, word_col_list, state_list]))
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]

    def __call__(self, pred):
        preds = np.array(pred)
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        texts = []
        scores = []
        for t in text:
            texts.append(t[0])
            scores.append(t[1])
        return (texts, scores)

class CTCLabelDecode(BaseRecLabelDecode):

    def __init__(self, character_list=None, use_space_char=True):
        super().__init__(character_list, use_space_char=use_space_char)

    def __call__(self, pred, return_word_box=False, **kwargs):
        preds = np.array(pred[0])
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True, return_word_box=return_word_box)
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs['wh_ratio_list'][rec_idx]
                max_wh_ratio = kwargs['max_wh_ratio']
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        texts = []
        scores = []
        for t in text:
            texts.append(t[0] if len(t) <= 2 else (t[0], t[2]))
            scores.append(t[1])
        return (texts, scores)

    def add_special_char(self, character_list):
        character_list = ['blank'] + character_list
        return character_list

class SortQuadBoxes:
    entities = 'SortQuadBoxes'

    def __init__(self):
        super().__init__()

    def __call__(self, dt_polys: List[np.ndarray]) -> np.ndarray:
        dt_boxes = np.array(dt_polys)
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)
        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and _boxes[j + 1][0][0] < _boxes[j][0][0]:
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

class CropByPolys:

    def get_minarea_rect_crop(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_a, index_b, index_c, index_d = (0, 1, 2, 3)
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2
        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = self.get_rotate_crop_image(img, np.array(box))
        return crop_img

    def get_rotate_crop_image(self, img: np.ndarray, points: list) -> np.ndarray:
        assert len(points) == 4, 'shape of points must be 4*2'
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
