# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from fvcore.common.timer import Timer

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

from .builtin_meta import _get_coco_instances_meta
from .lvis_v0_5_categories import LVIS_CATEGORIES as LVIS_V0_5_CATEGORIES
from .lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES

import torch
import numpy as np
from PIL import Image
"""
This file contains functions to parse LVIS-format annotations into dicts in the
"Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_lvis_json", "register_lvis_instances", "get_lvis_instances_meta"]


def register_lvis_instances(name, metadata, json_file, image_root, args=None):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    if args is not None:
        filter_open_cls = args['filter_open_cls']
        run_custom_img = args['run_custom_img']

    DatasetCatalog.register(name, lambda: load_lvis_json(json_file, image_root, name, \
        filter_open_cls=filter_open_cls, run_custom_img=run_custom_img))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="lvis", **metadata
    )


def load_lvis_json(json_file, image_root, dataset_name=None, filter_open_cls=False, clip_gt_crop=True, max_gt_per_img=500, run_custom_img=False):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        filter_open_cls: open-set setting, filter the rare (novel) categories during training
        clip_gt_crop: whether apply CLIP on GT regions; must filter images with empty annotations or too many GT bbox
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    if run_custom_img:
        return load_lvis_custom_img()
    
    from lvis import LVIS

    if 'train' in dataset_name and filter_open_cls: # openset training setting, filter the novel classes during training
        filter_open_cls = True
    else:
        filter_open_cls = False

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    if dataset_name is not None:
        meta = get_lvis_instances_meta(dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta)

    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file
    )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file))

    def get_file_name(img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(img_root + split_folder, file_name)

    dataset_dicts = []
    cls_type_dict = {cls_meta['id']: cls_meta['frequency'] for cls_meta in lvis_api.dataset['categories']} # map cls id to cls type
    area_dict = {'r': [], 'c': [], 'f': []}  # calculate box area for each type of class

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
            # variable will have a field with COCO-specific category mapping.
            if dataset_name is not None and "thing_dataset_id_to_contiguous_id" in meta:
                obj["category_id"] = meta["thing_dataset_id_to_contiguous_id"][anno["category_id"]]
            else:
                obj["category_id"] = anno["category_id"] - 1  # Convert 1-indexed to 0-indexed
            obj['frequency'] = cls_type_dict[anno["category_id"]]  # used for open-set filtering

            if filter_open_cls:  # filter categories for open-set training
                if obj['frequency'] == 'r':
                    continue
            area_dict[obj['frequency']].append(anno["bbox"][2] * anno["bbox"][3])

            segm = anno["segmentation"]  # list[list[float]]
            # filter out invalid polygons (< 3 points)
            valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            assert len(segm) == len(
                valid_segm
            ), "Annotation contains an invalid polygon with < 3 points"
            assert len(segm) > 0
            obj["segmentation"] = segm
            objs.append(obj)
        if (filter_open_cls or clip_gt_crop) and len(objs) == 0:  # no annotation for this image
            continue
        record["annotations"] = objs            
        dataset_dicts.append(record)
    
    # For the training in open-set setting, map original category id to new category id number (base categories)
    if filter_open_cls:  
        # get new category id in order
        old_to_new = {}  
        for i in range(len(cls_type_dict)):
            if cls_type_dict[i+1] != 'r': # cls_type_dict is 1-indexed
                old_to_new[i] = len(old_to_new)
        # map annotation to new category id
        for record in dataset_dicts:
            record.pop('not_exhaustive_category_ids')  # won't be used
            record.pop('neg_category_ids')  # won't be used
            for obj in record['annotations']:
                obj['category_id'] = old_to_new[obj['category_id']]  # 0-indexed id
                assert obj['frequency'] != 'r'
        logger.info("\n\nModel will be trained in the open-set setting! {} / {} categories are kept.\n".format(len(old_to_new),len(cls_type_dict)))

    # apply CLIP on GT regions: some images has large number of GT bbox (eg, 759), remove them, otherwise, OOM
    if clip_gt_crop:  
        dataset_dicts = sorted(dataset_dicts, key=lambda x: len(x["annotations"]), reverse=True)
        for record in dataset_dicts:
            record["annotations"] = record["annotations"][:max_gt_per_img]  # only <10 per 20k images in test have >300 GT boxes
    
    return dataset_dicts

def load_lvis_custom_img(custom_img_path='datasets/custom_images'):
    """
    This is a tentitive function for loading custom images.
    Given a folder of images (eg, 'datasets/custom_images'), load their meta data into a dictionary
    """
    custom_img_list = [os.path.join(custom_img_path, item) for item in os.listdir(custom_img_path)]

    dataset_dicts = []
    for f_i, file_name in enumerate(custom_img_list):
        record = {}
        record["file_name"] = file_name
        img_file = Image.open(record["file_name"])
        record["height"] = img_file.size[1]
        record["width"] = img_file.size[0]
        record["image_id"] = f_i
        
        dataset_dicts.append(record)
    
    return dataset_dicts

def get_lvis_instances_meta(dataset_name):
    """
    Load LVIS metadata.

    Args:
        dataset_name (str): LVIS dataset name without the split name (e.g., "lvis_v0.5").

    Returns:
        dict: LVIS metadata with keys: thing_classes
    """
    if "cocofied" in dataset_name:
        return _get_coco_instances_meta()
    if "v0.5" in dataset_name:
        return _get_lvis_instances_meta_v0_5()
    elif "v1" in dataset_name:
        return _get_lvis_instances_meta_v1()
    raise ValueError("No built-in metadata for dataset {}".format(dataset_name))


def _get_lvis_instances_meta_v0_5():
    assert len(LVIS_V0_5_CATEGORIES) == 1230
    cat_ids = [k["id"] for k in LVIS_V0_5_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V0_5_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


def _get_lvis_instances_meta_v1():
    assert len(LVIS_V1_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


if __name__ == "__main__":
    """
    Test the LVIS json dataset loader.

    Usage:
        python -m detectron2.data.datasets.lvis \
            path/to/json path/to/image_root dataset_name vis_limit
    """
    import sys
    import numpy as np
    from detectron2.utils.logger import setup_logger
    from PIL import Image
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from detectron2.utils.visualizer import Visualizer

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_lvis_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "lvis-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts[: int(sys.argv[4])]:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
