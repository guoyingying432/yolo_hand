import os
import numpy as np
import copy
import colorsys
import matplotlib.pyplot as plt
import cv2, imutils as im, argparse
from timeit import default_timer as timer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from nets.yolo3 import yolo_body, yolo_eval
from utils import letterbox_image
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.compat.v1.disable_v2_behavior()
# ---------------------------------------------------#
#   手部分类
# ---------------------------------------------------#
def CNNModel(image, model):
    im1 = image
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im1 = cv2.resize(im1, (64, 64))
    plt.imshow(im1)
    plt.show()
    t = []
    t.append(im1.reshape(64, 64))
    t = np.asarray(t)
    t = t.reshape(1, 64, 64, 1)
    res = model.predict(t)
    return res
# ---------------------------------------------------#
#   手部分割
# ---------------------------------------------------#
def hand_seg(img):
    CORRECTION_NEEDED = False
    # Define lower and upper bounds of skin areas in YCrCb colour space.
    lower = np.array([0, 139, 60], np.uint8)
    upper = np.array([255, 180, 127], np.uint8)
    # convert img into 300*x large
    r = 300.0 / img.shape[1]
    dim = (300, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    original = img.copy()

    # Extract skin areas from the image and apply thresholding
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    mask = cv2.inRange(ycrcb, lower, upper)
    skin = cv2.bitwise_and(ycrcb, ycrcb, mask=mask)
    _, black_and_white = cv2.threshold(mask, 127, 255, 0)

    # Find contours from the thresholded image
    _, contours, _ = cv2.findContours(black_and_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the maximum contour. It is usually the hand.
    length = len(contours)
    maxArea = -1
    final_Contour = np.zeros(img.shape, np.uint8)
    # print(final_Contour)
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        largest_contour = contours[ci]

    # print(largest_contour)
    # Draw it on the image, in case you need to see the ellipse.
    #cv2.drawContours(final_Contour, [largest_contour], 0, (0, 255, 0), 2)

    # Get the angle of inclination
    #ellipse = _, _, angle = cv2.fitEllipse(largest_contour)

    # original = cv2.bitwise_and(original, original, mask=black_and_white)

    # Vertical adjustment correction
    '''
    This variable is used when the result of hand segmentation is upside down. Will change it to 0 or 180 to correct the actual angle.
    The issue arises because the angle is returned only between 0 and 180, rather than 360.
    
    vertical_adjustment_correction = 0
    if CORRECTION_NEEDED: vertical_adjustment_correction = 180

    # Rotate the image to get hand upright
    if angle >= 90:
        black_and_white = im.rotate_bound(black_and_white, vertical_adjustment_correction + 180 - angle)
        original = im.rotate_bound(original, vertical_adjustment_correction + 180 - angle)
        final_Contour = im.rotate_bound(original, vertical_adjustment_correction + 180 - angle)
    else:
        black_and_white = im.rotate_bound(black_and_white, vertical_adjustment_correction - angle)
        original = im.rotate_bound(original, vertical_adjustment_correction - angle)
        final_Contour = im.rotate_bound(final_Contour, vertical_adjustment_correction - angle)
'''
    original = cv2.bitwise_and(original, original, mask=black_and_white)
    cv2.imshow('Extracted Hand', final_Contour)
    cv2.imshow('Original image', original)
    return original

    # Read image

# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
# --------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path": 'logs/last1.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/hand_classes.txt',
        "score": 0.5,
        "iou": 0.3,
        "model_image_size": (416, 416)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化yolo
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.__dict__.update(self._defaults)
        #self.class_names=[hand]
        self.class_names = self._get_class()
        #9*2先验框数字
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        start = timer()

        # 调整图片使其符合输入要求
        new_image_size = (self.model_image_size[0], self.model_image_size[1])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        small_pic = []

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 15
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            image1=np.array(image)
            image_cut=image1[left:right, top:bottom,:]
            cv2.imshow("image_cut",image_cut)
            image_cut_seg=hand_seg(image_cut)
            cv2.imshow("image_cut_seg",image_cut_seg)
            tf.keras.backend.clear_session()
            classify_model = load_model("./test.h5")
            print(CNNModel(image_cut_seg, classify_model))


            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()
