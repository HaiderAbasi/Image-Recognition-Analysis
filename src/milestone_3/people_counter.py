#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import time
from loguru import logger
import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from src.milestone_3 import args

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]



class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
        log = False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.log = log
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            
            # # filter output boxes to keep only persons
            # if outputs is not None and len(outputs) > 0:
            #     person_boxes = [] # list of detections[tensors]
            #     for output in outputs:
            #         if output is not None:
            #             mask = (output[:, 6] == 0)
            #             person_boxes.append(output[mask])
            #     outputs = person_boxes if len(person_boxes)>0 else [None]

            if self.log:
                logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, outputs, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if outputs is None:
            return img,[],"-"
        outputs = outputs.cpu()

        bboxes = outputs[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = outputs[:, 6]
        scores = outputs[:, 4] * outputs[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)

        # find the indices of non-person boxes
        non_person_idxs = torch.where(cls != 0)[0]

        if len(non_person_idxs) > 0:
            # compute size of each non-person bounding box
            non_person_sizes = (bboxes[non_person_idxs, 2] - bboxes[non_person_idxs, 0]) * (bboxes[non_person_idxs, 3] - bboxes[non_person_idxs, 1])

            # find the index of the largest non-person box
            largest_non_person_idx = non_person_idxs[torch.argmax(non_person_sizes)]

            # get class of object with largest non-person box
            largest_non_person_id = int(cls[largest_non_person_idx])
            focus_subject = self.cls_names[largest_non_person_id]

            #print("focus_subject:", focus_subject)
        else:
            focus_subject = "-"
            #print("No non-person boxes in image")
                
        people_boxes = [bboxes[idx].long().numpy() for idx, score in enumerate(scores) if score > cls_conf and cls[idx] == 0]
        # people_count = sum(score > cls_conf for score in scores)
        # if torch.is_tensor(people_count):
        #     people_count = int(people_count.item())
        # Convert tensor to numpy array
            
        return vis_res,people_boxes,focus_subject


class PeopleCounter():

    def __init__(self):
        self.log = False
        
        exp = get_exp(args.exp_file, args.name)
        
        if not args.experiment_name:
            args.experiment_name = exp.exp_name

        file_name = os.path.join(exp.output_dir, args.experiment_name)
        #os.makedirs(file_name, exist_ok=True)
        
        if self.log:
            logger.info("Args: {}".format(args))

        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

        model = exp.get_model()
        if self.log:
            logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        if args.device == "gpu":
            model.cuda()
            if args.fp16:
                model.half()  # to FP16
        model.eval()

        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        if self.log:
            logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        if self.log: 
            logger.info("loaded checkpoint done.")

        trt_file = None
        decoder = None

        self.predictor = Predictor(
            model, exp, COCO_CLASSES, trt_file, decoder,
            args.device, args.fp16, args.legacy,
        )
        self.draw = False         

    @staticmethod
    def get_image_list(path):
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
        return image_names

    def count(self, path_or_image):
        if isinstance(path_or_image, str):
            if os.path.isdir(path_or_image):
                files = self.get_image_list(path_or_image)
            else:
                files = [path_or_image]
        else:
            files = [path_or_image]
            
        files.sort()
        for image_name in files:
            outputs, img_info = self.predictor.inference(image_name)
            result_image,people_boxes,focus_subject = self.predictor.visual(outputs[0], img_info, self.predictor.confthre)
                
            if self.draw:
                for x,y,x2,y2 in people_boxes:
                    cv2.rectangle(result_image, (x,y), (x2,y2), (0, 255, 0), 2)
                
                
                cv2.imshow("detected",result_image)
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
                
        return people_boxes,focus_subject



if __name__ == "__main__":
    #args.nms =0.45
    #args.conf = 0.25

    pc = PeopleCounter()
    pc.count(args.path)
