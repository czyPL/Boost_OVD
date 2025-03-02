import json
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data.datasets.clip_prompt_utils import SimpleTokenizer,convert_example_to_features_bpe


def pre_tokenize(class_name):
    # tokenizer
    tokenizer = SimpleTokenizer()
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    input_ids = []
    for t1 in class_name:
        this_input_ids = convert_example_to_features_bpe(t1, tokenizer, sot_token, eot_token)
        input_ids.append(torch.tensor(this_input_ids, dtype=torch.long))

    input_ids = torch.stack(input_ids, 0)

    return input_ids


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.CLIP.CROP_REGION_TYPE = "GT"
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


def main(args):
    cfg = setup(args)

    # create model
    model = create_model(cfg)

    with open(args.concept_path, 'r') as f:
        concept_data = json.load(f)
    with open(args.anno_path, 'r') as f:
        ann_data = json.load(f)

    cats = ann_data['categories']
    cats = [(c['id'], c['name']) for c in cats]
    class_descriptions = [concept_data[c[1]] for c in cats]

    concept_feats = []
    with torch.no_grad():
        for class_description in class_descriptions:
            token_embedding = pre_tokenize(class_description).to(model.device) # [#des, context_length]
            text_features = model.lang_encoder.encode_text(token_embedding)
            # average over all templates
            text_features = text_features.mean(0, keepdim=True)
            concept_feats.append(text_features)

    concept_feats = torch.stack(concept_feats, 0)
    concept_feats = torch.squeeze(concept_feats).cpu()
    print(concept_feats.shape)
    torch.save(concept_feats, args.output_path)

    print("done!")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--concept-path",
        type=str,
        default="./ours/metadata/coco_ovd_llm_descriptions.json"
        # ./ours/metadata/coco_ovd_llm_descriptions.json ./ours/metadata/lvis_ovd_llm_descriptions.json
    )
    parser.add_argument(
        "--ann-path",
        type=str,
        default="./datasets/lvis/lvis_v1_val.json"
        # ./datasets/coco/annotations/ovd_ins_val2017_all.json ./datasets/lvis/lvis_v1_val.json
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./ours/metadata/coco_65_cls_emb_llm.pth"
        # ./ours/metadata/coco_65_cls_emb_llm.pth ./ours/metadata/lvis_1203_cls_emb_llm.pth _rn50x4
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