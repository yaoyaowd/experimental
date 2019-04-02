import argparse
import os
import sys
import numpy as np
import shutil
import urllib.request
import zipfile

import imgaug
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskutils

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

DATASET_YEAR = '2014' # or '2017'
DEFAULT_MODEL_DIR = os.path.join('/home/dong', 'model')
DEFAULT_LOG_DIR = os.path.join('/home/dong', 'logs')

class CocoConfig(Config):
    NAME = 'coco'
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 80

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DATASET_YEAR, class_ids=None, auto_download=False):
        if auto_download:
            self.auto_download(dataset_dir, subset, year)
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == 'minival' or subset == 'valminusminival':
            subset = 'val'
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        if not class_ids:
            class_ids = sorted(coco.getCatIds())
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            image_ids = list(set(image_ids))
        else:
            image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])
        for i in image_ids:
            self.add_image("coco", image_id=i,
                           path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                           width=coco.imgs[i]['width'],
                           height=coco.imgs[i]['height'],
                           annotations=coco.loadAnns(coco.getAnnIds(
                               imgIds=[i], catIds=class_ids, iscrowd=None)))

    def auto_download(self, data_dir, data_type, data_year):
        """
        Download the COCO dataset / annotation if requested.
        dataType: 'train', 'val', 'minival', 'valminusminival'
        dataYear: '2014', '2017
        """
        if data_type in ("minival", "valminusminival"):
            img_dir = "{}/{}{}".format(data_dir, "val", data_year)
            img_zip_file = "{}/{}{}.zip".format(data_dir, "val", data_year)
            img_url = "http://images.cocodataset.org/zips/{}{}.zip".format("val", data_year)
        else:
            img_dir = "{}/{}{}".format(data_dir, data_type, data_year)
            img_zip_file = "{}/{}{}.zip".format(data_dir, data_type, data_year)
            img_url = "http://images.cocodataset.org/zips/{}{}.zip".format(data_type, data_year)
        print("Image path: ", img_dir)
        print("Zip file: ", img_zip_file)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if not os.path.exists(img_zip_file):
            print("Download images from ", img_url)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            with urllib.request.urlopen(img_url) as resp, open(img_zip_file, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading")
            print("Unzipping " + img_zip_file)
            with zipfile.ZipFile(img_zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("... done unzipping")

        ann_dir = "{}/annotations".format(data_dir)
        if data_type == 'minival':
            ann_zip_file = "{}/instances_minival2014.json.zip".format(data_dir)
            ann_url = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unzip_dir = ann_dir
        elif data_type == "valminusminival":
            ann_zip_file = "{}/instances_valminusminival2014.json.zip".format(data_dir)
            ann_url = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unzip_dir = ann_dir
        else:
            ann_zip_file = "{}/annotations_trainval{}.zip".format(data_dir, data_year)
            ann_url = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(data_year)
            unzip_dir = data_dir

        if not os.path.exists(ann_dir):
            os.makedirs(ann_dir)
        if not os.path.exists(ann_zip_file):
            print("Downloading annotations from ", ann_url)
            with urllib.request.urlopen(ann_url) as resp, open(ann_zip_file, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + ann_zip_file)
            with zipfile.ZipFile(ann_zip_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            print("... done unzipping")

        print("Image dir: ", img_dir)
        print("Annotation dir: ", unzip_dir)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != 'coco':
            return super(CocoDataset, self).load_mask(image_id)
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        for annotation in annotations:
            class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.ann2mask(annotation, info['height'], info['width'])
                if m.max() < 1:
                    continue
                if annotation['iscrowd']:
                    class_id *= -1
                    # For crowd masks, ann2mask sometimes returns a mask smaller than the
                    # given dimensions. If so, resize it.
                    if m.shape[0] != info['height'] or m.shape[1] != info['width']:
                        m = np.ones([info['height'], info['width']], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'coco':
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    def ann2rle(self, ann, height, width):
        """Annotation to run length encoding."""
        segm = ann['segmentation']
        if isinstance(segm, list):
            rles = maskutils.frPyObjects(segm, height, width)
            rle = maskutils.merge(rles)
        elif isinstance(segm['counts'], list):
            rle = maskutils.frPyObjects(segm, height, width)
        else:
            rle = segm
        return rle

    def ann2mask(self, ann, height, width):
        rle = self.ann2rle(ann, height, width)
        m = maskutils.decode(rle)
        return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', metavar='<command>', help='`train` or `evaluation`')
    parser.add_argument('--dataset', default='/home/dong/coco',
                        required=False, metavar='/path/to/coco/')
    parser.add_argument('--model', required=False,
                        default=DEFAULT_MODEL_DIR, metavar='/path/to/weight.h5')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOG_DIR, metavar='/path/to/logs')
    parser.add_argument('--limit', required=False, default=500, help='Images for evaluation')
    args = parser.parse_args()

    print("Command:", args.command)
    print("Dataset:", args.dataset)
    print("Model:", args.model)
    print("Logs:", args.logs)
    print("Limit:", args.limit)

    config = CocoConfig()
    if args.command != 'train':
        config.GPU_COUNT = 1
        config.IMAGES_PER_GPU = 1
        config.DETECTION_MIN_CONFIDENCE = 0
    config.display()

    if args.command == 'train':
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "valminusminival", year=DATASET_YEAR)
        dataset_train.prepare()

        dataset_val = CocoDataset()
        val_type = "val" if DATASET_YEAR in '2017' else 'minival'
        dataset_val.load_coco(args.dataset, val_type, year=DATASET_YEAR)
        dataset_val.prepare()

        augmentation = imgaug.augmenters.Fliplr(0.5)

        print("Train network heads")
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE,
                    epochs=40, layers='heads', augmentation=augmentation)
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE,
                    epochs=120, layers='4+', augmentation=augmentation)
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10,
                    epochs=160, layers='all', augmentation=augmentation)
