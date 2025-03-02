import argparse
import torch
from torch.nn import functional as F

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt-path",
        type=str,
        default="./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth"
        # "./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth" or "./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth"  _rn50x4
    )
    parser.add_argument(
        "--llm-path",
        type=str,
        default="./ours/metadata/coco_65_cls_emb_llm.pth"
        # "./ours/metadata/coco_65_cls_emb_llm.pth" or "./ours/metadata/lvis_1203_cls_emb_llm.pth"  _rn50x4
    )
    parser.add_argument(
        "--mllm-path",
        type=str,
        default="./ours/metadata/coco_65_cls_emb_mllm.pth"
        # "./ours/metadata/coco_65_cls_emb_mllm.pth" or "./ours/metadata/lvis_1203_cls_emb_mllm.pth"  _rn50x4
    )
    parser.add_argument(
        "--img-path",
        type=str,
        default="./ours/metadata/coco_65_cls_emb_img.pth"
        # "./ours/metadata/coco_65_cls_emb_img.pth" or "./ours/metadata/lvis_1203_cls_emb_img.pth"  _rn50x4
    )
    parser.add_argument(
        "--pt-weight",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--mllm-weight",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--img-weight",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco" # "coco" "lvis"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./ours/metadata/coco_65_cls_emb_mean4.pth"
        # "./ours/metadata/coco_65_cls_emb_mean4.pth" "./ours/metadata/lvis_1203_cls_emb_mean4.pth"
    )
    parser.add_argument(
        "--base-save-path",
        type=str,
        default="./ours/metadata/coco_48_cls_emb_mean4.pth"
        # "./ours/metadata/coco_48_base_cls_emb_mean4.pth" "./ours/metadata/lvis_866_base_cls_emb_mean4.pth"
    )
    return parser

def main(args):
    pt_emb = F.normalize(torch.load(args.pt_path), p=2.0, dim=-1)
    llm_emb = F.normalize(torch.load(args.llm_path), p=2.0, dim=-1)
    mllm_emb = F.normalize(torch.load(args.mllm_path), p=2.0, dim=-1)
    img_emb = F.normalize(torch.load(args.img_path), p=2.0, dim=-1)

    # Merge
    combine_emb = torch.mean(torch.stack([args.pt_weight * pt_emb, args.llm_weight * llm_emb, \
                                          args.mllm_weight * mllm_emb, args.imh_weight * img_emb]), dim=0)
    print(combine_emb.shape)
    torch.save(combine_emb, args.save_path)

    if args.dataset == 'coco':
        from detectron2.data.datasets.coco_zeroshot_categories import COCO_SEEN_CLS,COCO_OVD_ALL_CLS
        base2all = {COCO_SEEN_CLS.index(cat):COCO_OVD_ALL_CLS.index(cat) for cat in COCO_SEEN_CLS}
    else:
        from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
        lvis_base_cats = sorted([cat for cat in LVIS_CATEGORIES if cat['frequency'] != 'r'], key=lambda x: x['id'])
        base2all = {base_id : cat['id'] - 1 for base_id,cat in enumerate(lvis_base_cats)}


    # Get Base Emb
    base_emb = []
    for index in range(len(base2all)):
        base_emb.append(combine_emb[base2all[index]])
    base_emb = torch.stack(base_emb)

    print(base_emb.shape)
    torch.save(base_emb, args.base_save_path)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)