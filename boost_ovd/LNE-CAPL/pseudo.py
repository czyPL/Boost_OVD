'''
    pseudo COCO or LVIS unclassified in COCO format
'''
import argparse
import json, os
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict
from supervision import Detections
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-root",
        type=str,
        default="./datasets/coco/"
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        default="./datasets/coco/annotations/ovd_ins_train2017_b.json"
        # "./datasets/coco/annotations/ovd_ins_train2017_b.json" "./datasets/lvis/lvis_v1_train_norare.json"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./datasets/coco/annotations/ovd_ins_train2017_b_unclassified_pseudo.json"
        # "./datasets/coco/annotations/ovd_ins_train2017_b_unclassified_pseudo.json" "./datasets/lvis/lvis_v1_train_norare_unclassified_pseudo.json"
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="object"
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="./ours/LNE-CAPL/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./ours/LNE-CAPL/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
    parser.add_argument(
        "--is-coco",
        type=bool,
        default=False  # True
    )
    parser.add_argument(
        "--allimg-gt-path",
        type=str,
        default="./datasets/coco/annotations/ovd_ins_train2017_all.json"
    )
    return parser


def xyxy2xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def xywh2xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h])


def ncxcywh2xyxy(boxs, img_w, img_h):
    new_boxs = []
    for box in boxs:
        ncx, ncy, nw, nh = box
        new_boxs.append(
            [max(0, (ncx - 0.5 * nw) * img_w), max(0, (ncy - 0.5 * nh) * img_h), min(img_w, (ncx + 0.5 * nw) * img_w),
             min(img_h, (ncy + 0.5 * nh) * img_h)])
    return np.array(new_boxs)


@dataclass
class Detections2(Detections):
    @classmethod
    def from_groundingdino(cls, groundingdino_results, img_w, img_h) -> Detections:
        boxes, logits, _ = groundingdino_results

        return cls(
            xyxy=ncxcywh2xyxy(boxes.numpy(), img_w, img_h),
            confidence=logits.numpy(),
            class_id=np.array([1] * len(logits))
        )

    @classmethod
    def from_coco(cls, results) -> Detections:
        xyxys = []
        for result in results:
            xyxys.append(xywh2xyxy(result['bbox']))

        return cls(
            xyxy=np.array(xyxys),
            confidence=np.array([1] * len(xyxys)),
            class_id=np.array([1] * len(xyxys)),
        )


def pseudo_img(img_path, gt_annos, model, text_prompt, box_threshold, text_threshold, device):
    image_source, image = load_image(img_path)
    img_h, img_w, _ = image_source.shape
    img_area = img_h * img_w

    # Pseudo ncxcywh
    results = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )
    # Self processing
    pseudo_detections = Detections2.from_groundingdino(results, img_w, img_h)  # change to supervision format
    pseudo_detections = pseudo_detections.with_nms(threshold=0.5, class_agnostic=True)  # NMS
    pseudo_detections = pseudo_detections[(pseudo_detections.area / img_area) < 0.95]  # Too Large
    # Merge processing by GT
    if len(gt_annos) != 0:
        gt_detections = Detections2.from_coco(gt_annos)
        # Merge processing
        merge_detections = Detections.merge([gt_detections, pseudo_detections])
        merge_detections = merge_detections.with_nms(threshold=0.7, class_agnostic=True)  # NMS (GT's score is 1)
        # Remain pseudo
        remain_ids = (pseudo_detections.xyxy[:, None] == merge_detections.xyxy).all(-1).any(1)  # Intersection by Rows
        pseudo_boxs = pseudo_detections.xyxy[remain_ids]  # n x 4 xyxy
    else:
        pseudo_boxs = pseudo_detections.xyxy

    if len(pseudo_boxs) == 0:
        return None

    return pseudo_boxs  # numpy: n x 4 xyxy


def main(args):
    model = load_model(args.model_config_path, args.model_path)

    with open(args.gt_path, 'r') as f:
        origin_gt = json.load(f)

    # 1.Change Image
    if args.is_coco:
        with open(args.allimg_gt_path, 'r') as f:
            allimg_gt = json.load(f)
        origin_gt['images'] = allimg_gt['images']  # image_num > b_image_num
        del allimg_gt
    else:
        for img in origin_gt['images']:
            img['not_exhaustive_category_ids'] = []
            img['neg_category_ids'] = []
            # add 'file_name'
            img['file_name'] = img['coco_url'].replace('http://images.cocodataset.org/', '')

    # 2.Change Anns
    for ann in origin_gt['annotations']:
        ann['category_id'] = 1

    origin_ann_id_max = max([i['id'] for i in origin_gt['annotations']])

    imgid2paths = {i['id']: os.path.join(args.img_root, i['coco_url'].replace('http://images.cocodataset.org/', '')) for
                   i in origin_gt['images']}
    print("Image Num:", len(imgid2paths))

    imgid2annos = defaultdict(list)
    for i in origin_gt['annotations']:
        imgid2annos[i['image_id']].append(i)
    print("Origin ann num:", len(origin_gt['annotations']))

    new_ann_id = origin_ann_id_max
    for imgid, imgpath in tqdm(imgid2paths.items()):
        anns = imgid2annos[imgid]  # if imgid not in imgid2annos get []
        # pseudo
        pseudo_boxs = pseudo_img(imgpath, anns, model, args.text_prompt, args.box_threshold, args.text_threshold,
                                 args.device)  # numpy: n x 4 xyxy
        if pseudo_boxs is None:
            continue
        for pseudo_box in pseudo_boxs:
            new_ann_id += 1
            box = xyxy2xywh(pseudo_box)
            pseudo = {
                "id": new_ann_id,
                "image_id": imgid,
                "category_id": 1,
                "area": int(box[2] * box[3]),
                "segmentation": [],
                "bbox": box,
                "iscrowd": 0
            }

            origin_gt['annotations'].append(pseudo)

    print("Final ann num:", len(origin_gt['annotations']))

    # 3.Change Class
    if args.is_coco:
        origin_gt['categories'] = [{
            'name': 'object',
            'id': 1
        }]
    else:
        origin_gt['categories'] = [{
            'name': 'object',
            'instance_count': len(origin_gt['annotations']),
            'def': 'object',
            'synonyms': ['object'],
            'image_count': len(origin_gt['images']),
            'id': 1,
            'frequency': 'f',
            'synset': 'object'
        }]

    # 4.Save
    with open(args.save_path, 'w') as f:
        f.write(json.dumps(origin_gt))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
