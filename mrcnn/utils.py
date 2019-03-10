"""
A list of util functions for Mask R-CNN.
Shameless copied and modified from:
https://github.com/matterport/Mask_RCNN
"""

import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil

COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


def extract_bbox(mask):
    """
    Compute bounding boxes from masks.
    masks: [height, width, num_instances] mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """
    Calculate the intersection of union of the given box with the array of the given boxes.
    :param box: 1D vector [y1, x1, y2, x2]
    :param boxes: [boxes_count, (y1, x1, y2, x2)]
    :param box_area: float, the area of the box
    :param boxes_area: array of length boxes count
    :return:
    """
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = boxes_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and combines the results.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)
    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]
    return result


def box_refinement_graph(box, gt_box):
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)
    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width
    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)
    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


# Anchors

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    :param scales: 1D array of anchor sizes in pixels, example: [32, 64, 128]
    :param ratios: 1D array of anchor ratios of width/height, example: [0.5, 1, 2]
    :param shape: [height, width]
    :param feature_stride: stride of the feature map
    :param anchor_stride: stride of anchors on the feature map.
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
    return boxes

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of pyramid, but each ratio is used in all levels
    of the pyramid"""
    # TODO(dwang) understand this, smaller feature shape with large scales.
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


# Parse Image Meta

def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components."""
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[: 7:11]
    scale =  meta[: 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


# Miscellenous Graph Functions.

def trim_zeros_graph(boxes, name='trime_zeros'):
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def batch_pack_graph(x, counts, num_rows):
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def norm_boxes(boxes, shape):
    h, w = shape
    scale = np.array([h-1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)

def denorm_boxes(boxes, shape):
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)

def norm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)

def denorm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)


# Dataset Class

class Dataset(object):
    """
    The base class for dataset class.
    class DatasetName(Dataset):
        def load_cats_and_dogs(self):
            pass
        def load_mask(self, image_id):
            pass
        def image_reference(self, image_id):
            pass
    """
    def __init__(self, class_map=None):
        self.image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_info = {}

    def add_class(self, source, class_id, class_name):
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                return
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        return ""

    def prepare(self, class_map=None):
        def clean_name(name):
            return ",".join(name.split(",")[:1])

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info["source"], info["id"]): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info["source"], info["id"]): id
                                      for info, id in zip(self.image_info, self.image_ids)}
        self.sources = list(set(info["source"] for info in self.class_info))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = []
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info["source"]:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example: dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def source_image_link(self, image_id):
        """Returns the path or URL to the image"""
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        image = skimage.io.imread(self.source_image_link(image_id))
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if self.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this metho to load
        instance masks and return them in the form of an array of binary masks of shape
        [height, width, instances]
        """
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids
