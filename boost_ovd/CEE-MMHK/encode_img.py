import torch
import copy

from detectron2.config import configurable
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data.datasets.coco import register_coco_instances

from detectron2.data import build_detection_test_loader, DatasetMapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from collections import defaultdict

import numpy as np

class OurDatasetMapper(DatasetMapper):
    @configurable
    def __init__(self, *, augmentations_obj, augmentations_img, **kwargs):
        super().__init__(**kwargs)
        self.augmentations_obj = T.AugmentationList(augmentations_obj)
        self.augmentations_img = T.AugmentationList(augmentations_img)

    @classmethod
    def from_config(cls, cfg, is_train: bool = False):
        ret = super().from_config(cfg, is_train)
        obj_size = cfg.INPUT.OBJ_SIZE
        img_size = cfg.INPUT.IMG_SIZE
        ret['augmentations_obj'] = [T.ResizeShortestEdge(obj_size, obj_size, "choice")]
        ret['augmentations_img'] = [T.ResizeShortestEdge(img_size, img_size, "choice")]
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)

        # Distinguish between images and objects
        if "train2017" in dataset_dict["file_name"]:
            transforms = self.augmentations_obj(aug_input)
        else:
            transforms = self.augmentations_img(aug_input)

        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            if self.clip_crop:  # still load the GT annotations
                pass
            else:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)
                return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.CLIP.CROP_REGION_TYPE = "GT"
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.INPUT.OBJ_SIZE = args.obj_size
    cfg.INPUT.IMG_SIZE = args.img_size
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def create_model(cfg):
    # create model
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    assert model.use_clip_c4  # use C4 + resnet weights from CLIP
    assert model.use_clip_attpool  # use att_pool from CLIP to match dimension

    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model

def extract_region_feats(model, batched_inputs):
    # model inference
    # 1. localization branch: get the gts
    cls_ids = []
    proposals = []
    for r_i, b_input in enumerate(batched_inputs):
        this_gt = copy.deepcopy(b_input["instances"])  # Instance
        gt_boxes = this_gt._fields['gt_boxes'].to(model.device)
        gt_classes = this_gt._fields['gt_classes'].tolist()
        this_gt._fields = {'proposal_boxes': gt_boxes,
                           'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to(model.device)}
        proposals.append(this_gt)
        cls_ids.extend(gt_classes)

    # 2. recognition branch: get 2D feature maps using the backbone of recognition branch
    images = model.preprocess_image(batched_inputs) # [B, C, H, W]
    features = model.backbone(images.tensor)

    # 3. given the gts, crop region features from 2D image features
    proposal_boxes = [x.proposal_boxes for x in proposals]
    box_features = model.roi_heads._shared_roi_transform(
        [features[f] for f in model.roi_heads.in_features], proposal_boxes, model.backbone.layer4
    ) # [B x #boxes, 2048, 7, 7] B=1

    att_feats = model.backbone.attnpool(box_features)  # region features
    region_feats = att_feats.cpu()  # region features, [#boxes, d]
    return cls_ids,region_feats

def main(args):
    cfg = setup(args)

    # create model
    model = create_model(cfg)

    # register sample image
    sample_data_name = "sample_img"
    register_coco_instances(
        sample_data_name,
        {},  # empty metadata, it will be overwritten in load_coco_json() function
        args.example_path,
        args.img_root
    )
    data_loader = build_detection_test_loader(cfg,sample_data_name,mapper=OurDatasetMapper(cfg,False))

    clsid2features = defaultdict(list) # {id:[features]}
    # process each image
    for i, batched_inputs in enumerate(data_loader):
        # extract region features
        with torch.no_grad():
            cls_ids,region_feats = extract_region_feats(model, batched_inputs)
        for cls_id,region_feat in zip(cls_ids,region_feats):
            clsid2features[cls_id].append(region_feat)

    if args.mean:
        for clsid,features in clsid2features.items():
            clsid2features[clsid] = torch.mean(torch.stack(features), dim=0)
        img_feats = torch.stack(list(dict(sorted(clsid2features.items())).values()),dim=0) # cls_num * emb_dim
        print(img_feats.shape)
    else:
        img_feats = {id:torch.stack(features,dim=0) for id,features in clsid2features.items()} # {cls_id:tensor(region_num,emb_dim)}
        print(len(img_feats))

    print(args.output_path)
    torch.save(img_feats, args.output_path)

    print("done!")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--example-path",
        type=str,
        default="./ours/metedata/coco_ovd_image_examples.json"
        # "./ours/metedata/coco_ovd_image_examples.json" "./ours/metedata/lvis_ovd_image_examples.json"
    )
    parser.add_argument(
        "--img-root",
        type=str,
        default="./datasets/coco/train2017"
        # COCO:"./datasets/coco/train2017" LVIS:"./datasets/coco"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./ours/metadata/coco_65_cls_emb_img.pth"
        # "./ours/metadata/coco_65_cls_emb_img.pth" "./ours/metadata/lvis_1203_cls_emb_img.pth" _rn50x4
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--obj-size",
        type=int,
        default=800
    )
    parser.add_argument(
        "--mean",
        type=bool,
        default=True
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
