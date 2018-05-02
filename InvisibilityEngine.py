"""
The functionality of the code in this file is the same as the code found in the tutorial_single_image Notebook.

Written by Michael Nation 5/2/2018
"""

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mrcnn.model as modellib
import numpy as np

from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from mrcnn import coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InvisibilityEngine():

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        self.seg_model = self.init_instance_segmentation_model()
        self.sunglasses_model = InceptionV3(weights='imagenet')

    def init_instance_segmentation_model(self):
        config = InferenceConfig()

        seg_model = modellib.MaskRCNN(mode='inference', model_dir='', config=config)
        seg_model.load_weights('mask_rcnn_coco.h5', by_name=True)

        return seg_model

    def person_has_sunglasses(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_rgb, (299, 299))
        img_resized = img_to_array(img_resized)
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = preprocess_input(img_resized)

        preds = self.sunglasses_model.predict(img_resized)
        preds_decoded = imagenet_utils.decode_predictions(preds)

        has_sunglasses = False
        for imagenetID, label, prob in preds_decoded[0][:2]:
            if label == 'sunglass' or label == 'sunglasses':
                has_sunglasses = True
                break

        return has_sunglasses

    def get_bbox_of_face(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cascade_file_path = 'haarcascade_frontalface_default.xml'
        cascade_clf = cv2.CascadeClassifier(cascade_file_path)

        faces = cascade_clf.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

        return faces[0] if len(faces) > 0 else None

    def run_invisibility_on_image(self, img, img_background):
        """
        Args:
            img (numpy.array): RGB image. Contains people. Used as "read-only" and will not be modified by this function.
            img_background (numpy.array): RGB image. There are no people in this image.
        """

        # img_copy will be modified with the pixel manipulations and returned by the function
        img_copy = np.copy(img)

        results = self.seg_model.detect([img], verbose=0)
        results = results[0]

        # shape of the masks are (h, w, object_ix)
        n_objects_found = len(results['class_ids'])
        for ix in range(n_objects_found):
            if results['class_ids'][ix] == self.class_names.index('person'):
                bbox = self.get_bbox_of_face(img)
                if bbox is not None:
                    x, y, w, h = bbox
                    img_face = img[y: y + h, x: x + w]
                    if self.person_has_sunglasses(img_face):
                        mask = self.get_person_mask(results['masks'], ix)
                        img_copy[mask] = img_background[mask]

        return img_copy

    def get_person_mask(self, masks, ix_person):
        p_mask = masks[:, :, ix_person]

        padding = 10
        p_mask_2 = np.roll(p_mask, padding, axis=0)
        p_mask_3 = np.roll(p_mask, -padding, axis=0)
        p_mask_4 = np.roll(p_mask, padding, axis=1)
        p_mask_5 = np.roll(p_mask, -padding, axis=1)
        p_masks = [p_mask, p_mask_2, p_mask_3, p_mask_4, p_mask_5]
        p_masks_merged = np.logical_or.reduce(p_masks, axis=0)

        other_masks = np.delete(masks, ix_person, axis=2)
        other_masks_merged = np.logical_or.reduce(other_masks, axis=2)

        person_mask_minus_others = p_masks_merged & np.logical_not(other_masks_merged)

        return person_mask_minus_others