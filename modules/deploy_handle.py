import argparse
import sys
import numpy as np
import base64
import json
import cv2
import os
import ctypes
import platform
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from interface.python.interface import AisDeployC


class DeployHandle(object):
    def __init__(
            self,
            model_path,
            language,
            lib_path,
            gpu_id=0


    ):
        self.model_path = model_path
        self.language = language
        self.deploy_obj = AisDeployC(lib_path)
        self.gpu_id = gpu_id


    def model_init(self):

        ret = self.deploy_obj.model_initialize(self.model_path, self.gpu_id)
        if ret != 0:
            if self.language == "Chinese":
                return dict(
                    code=-1,
                    msg="模型初始化失败！"
                )
            elif self.language == "English":
                return dict(
                    code=-1,
                    msg="Fail to init models！"
                )
        return ret

    def deploy(self, image):
        if image is None:
            print("[ERROR] image is None")
            return dict(
                code=-1,
                msg="图像为空"
            )
        if not os.path.exists(self.model_path):
            print("[ERROR] self.model_path not exist!")
            return dict(
                code=-1,
                msg="模型路径不存在"
            )

        encoded_image= cv2.imencode(".jpg", image.copy())[1].tobytes()  # bytes类型

        input_json = {"data_list": []}
        qrcode = base64.b64encode(encoded_image).decode()
        # f= open("D:\TheWorkData\BandCLocationUi\image_save\puma.jpg", 'rb')
        # qrcode = base64.b64encode(f.read()).decode()

        file_json = {"type": "base64", "data": qrcode, "ch": 3}
        input_json["data_list"].append(file_json)
        try:
            ret_val = self.deploy_obj.process(input_json)
        except:
            return dict(
                code=-1,
                msg="模型推理失败！",
                data=None

            )
        if ret_val is None:
            return dict(
                code=0,
                msg="模型推理成功！未找到对象！",
                data=None

            )

        return dict(
            code=0,
            msg="模型推理成功！",
            data=ret_val
        )
    def deployjson(self, input_json):
        try:
            ret_val = self.deploy_obj.process(input_json)
        except:
            return dict(
                code=-1,
                msg="模型推理失败！",
                data=None

            )
        if ret_val is None:
            return dict(
                code=0,
                msg="模型推理成功！未找到对象！",
                data=None

            )

        return dict(
            code=0,
            msg="模型推理成功！",
            data=ret_val
        )

    def overlap_filter(
            self,
            result,
            scope_coord,
            overlap_threshold=0.5
    ):
        # overlap is the intersection area of two rectangles divided by the area of the smaller one
        if result is None:
            print("[ERROR] result_list is None")
            return list()
        if len(result) == 0:
            print("[ERROR] result_list is empty")
            return list()
        if overlap_threshold < 0 or overlap_threshold > 1:
            print("[ERROR] overlap_threshold is invalid")
            return list()
        (s_x1, s_y1, s_x2, s_y2) = scope_coord
        if 'rotated_bbox' in str(result):
            x = int(result["rotated_bbox"][0])
            y = int(result["rotated_bbox"][1])
            w = int(result["rotated_bbox"][2])
            h = int(result["rotated_bbox"][3])

        else:
            x = int(result["bbox"][0])
            y = int(result["bbox"][1])
            w = int(result["bbox"][2]-result["bbox"][0])
            h = int(result["bbox"][3]-result["bbox"][1])
        score = result["score"]

        # ((x_center, y_center), (w, h), theta)

        # calculate overlap area with scope in cv2
        rect1 = ((x, y), (w, h), 0)
        s_x_center = int((s_x1 + s_x2) / 2)
        s_y_center = int((s_y1 + s_y2) / 2)
        s_w = int(s_x2 - s_x1)
        s_h = int(s_y2 - s_y1)
        rect2 = ((s_x_center, s_y_center), (s_w, s_h), 0)
        box1 = cv2.boxPoints(rect1)
        box2 = cv2.boxPoints(rect2)
        box1 = np.int0(box1)
        box2 = np.int0(box2)
        overlap = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
        if overlap is not None:
            overlap_area = cv2.contourArea(overlap)
            box1_area = cv2.contourArea(box1)
            box2_area = cv2.contourArea(box2)
            if box1_area < box2_area:
                smaller_area = box1_area
            else:
                smaller_area = box2_area
            overlap = overlap_area / smaller_area
        else:
            overlap = 0

        if overlap >= overlap_threshold:
            return result
        else:
            return list()

    def cal_delta_offset(
            self,
            result,
            scope_coord
    ):
        if result is None:
            print("[ERROR] result is None")
            return None
        if len(result) == 0:
            print("[ERROR] result is empty")
            return list()
        (s_x1, s_y1, s_x2, s_y2) = scope_coord
        if 'rotated_bbox' in str(result):
             x = int(result["rotated_bbox"][0])
             y = int(result["rotated_bbox"][1])


        else:
            x = int(result["bbox"][0])
            y = int(result["bbox"][1])

        # ((x_center, y_center), (w, h), theta)
        s_x_center = int((s_x1 + s_x2) / 2)
        s_y_center = int((s_y1 + s_y2) / 2)
        delta_x = x - s_x_center
        delta_y = y - s_y_center

        delta_d = np.sqrt(delta_x ** 2 + delta_y ** 2)

        result["delta_x"] = delta_x
        result["delta_y"] = delta_y
        result["delta_d"] = delta_d

        return result