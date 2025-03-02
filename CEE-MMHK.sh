# 1. Acquire External Knowledge

# ---------------Generate LLM Category Descriptions---------------
# COCO
python3 ./boost_ovd/CEE-MMHK/llm.py \
--ann-path ./datasets/coco/annotations/ovd_ins_val2017_all.json \
--output-path ./boost_ovd/metadata/coco_ovd_llm_descriptions.json \
--openai-model gpt-3.5-turbo-instruct

# LVIS
python3 ./boost_ovd/CEE-MMHK/llm.py \
--ann-path ./datasets/lvis/lvis_v1_val.json \
--output-path ./boost_ovd/metadata/lvis_ovd_llm_descriptions.json \
--openai-model gpt-3.5-turbo-instruct

# -----------------------Get Image Examples-----------------------
# COCO
python3 ./boost_ovd/CEE-MMHK/img.py \
--ann-path ./datasets/coco/annotations/ovd_ins_val2017_all.json \
--use-t2i False \
--t2i-root None \
--output-path ./boost_ovd/metadata/coco_ovd_image_examples.json \
--base-num 5 \
--novel-num 5 \
--min-size 4096

# LVIS
python3 ./boost_ovd/CEE-MMHK/img.py \
--ann-path ./datasets/lvis/lvis_v1_train.json \
--use-t2i True \
--t2i-root ./boost_ovd/metadata/lvis_ovd_t2i_imgs \
--K 5 --openai-model dall-e-2 --img-size 512x512 \
--output-path ./boost_ovd/metadata/coco_ovd_image_examples.json \
--base-num 5 \
--novel-num 5 \
--min-size 512

# ---------------Generate MLLM Category Descriptions---------------
# COCO
python3 ./boost_ovd/CEE-MMHK/mllm.py \
--ann-path ./datasets/coco/annotations/ovd_ins_train2017_all.json \
--crop_path ./datasets/crop/coco_crop_150_5 \
--output-path ./boost_ovd/metadata/coco_ovd_mllm_descriptions.json \
--base-crop-num 150 \
--novel-crop-num 5 \
--extra-data None

# LVIS
python3 ./boost_ovd/CEE-MMHK/mllm.py \
--ann-path ./datasets/lvis/lvis_v1_train.json \
--crop_path ./datasets/crop/lvis_crop_150_5 \
--output-path ./boost_ovd/metadata/lvis_ovd_mllm_descriptions.json \
--base-crop-num 150 \
--novel-crop-num 5 \
--extra-data ./our/metadata/lvis_ovd_t2i_imgs


# 2. Encode External Knowledge

# --------------------------------------------LLM/MLLM-------------------------------------
# RN50
# COCO
python3 ./boost_ovd/CEE-MMHK/encode_text.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.GET_CONCEPT_EMB True \
--concept-path ./boost_ovd/metadata/coco_ovd_{llm/mllm}_descriptions.json \
--ann-path ./datasets/coco/annotations/ovd_ins_val2017_all.json \
--output-path ./boost_ovd/metadata/coco_65_cls_emb_{llm/mllm}.pth

# LVIS
python3 ./boost_ovd/CEE-MMHK/encode_text.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.GET_CONCEPT_EMB True \
--concept-path ./boost_ovd/metadata/lvis_ovd_{llm/mllm}_descriptions.json \
--ann-path ./datasets/lvis/lvis_v1_val.json \
--output-path ./boost_ovd/metadata/lvis_1203_cls_emb_{llm/mllm}.pth

# RN50x4
# COCO
python3 ./boost_ovd/CEE-MMHK/encode_text.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.CLIP.GET_CONCEPT_EMB True \
--concept-path ./boost_ovd/metadata/coco_ovd_{llm/mllm}_descriptions.json \
--ann-path ./datasets/coco/annotations/ovd_ins_val2017_all.json \
--output-path ./boost_ovd/metadata/coco_65_cls_emb_{llm/mllm}_rn50x4.pth

# LVIS
python3 ./boost_ovd/CEE-MMHK/encode_text.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.CLIP.GET_CONCEPT_EMB True \
--concept-path ./boost_ovd/metadata/lvis_ovd_{llm/mllm}_descriptions.json \
--ann-path ./datasets/lvis/lvis_v1_val.json \
--output-path ./boost_ovd/metadata/lvis_1203_cls_emb_{llm/mllm}_rn50x4.pth

# -----------------------------------IMG--------------------------------------
# RN50
# COCO
python3 ./boost_ovd/CEE-MMHK/encode_img.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
--example-path ./boost_ovd/metedata/coco_ovd_image_examples.json \
--img-root ./datasets/coco/train2017 \
--output-path ./boost_ovd/metadata/coco_65_cls_emb_img.pth \

# LVIS
python3 ./boost_ovd/CEE-MMHK/encode_img.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
--example-path ./boost_ovd/metedata/lvis_ovd_image_examples.json \
--img-root ./datasets/coco \
--output-path ./boost_ovd/metadata/lvis_1203_cls_emb_img.pth \

# RN50x4
# COCO
python3 ./boost_ovd/CEE-MMHK/encode_img.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
--example-path ./boost_ovd/metedata/coco_ovd_image_examples.json \
--img-root ./datasets/coco/train2017 \
--output-path ./boost_ovd/metadata/coco_65_cls_emb_img_rn50x4.pth \

# LVIS
python3 ./boost_ovd/CEE-MMHK/encode_img.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
--example-path ./boost_ovd/metedata/lvis_ovd_image_examples.json \
--img-root ./datasets/coco \
--output-path ./boost_ovd/metadata/lvis_1203_cls_emb_img_rn50x4.pth \


# 3. Merge four types of embedding
# COCO
python3 ./boost_ovd/CEE-MMHK/merge_embed.py \
--pt-path ./pretrained_ckpt/concept_emb/coco_65_cls_emb{_rn50x4}.pth \
--llm-path ./boost_ovd/metadata/coco_65_cls_emb_llm{_rn50x4}.pth \
--mllm-path ./boost_ovd/metadata/coco_65_cls_emb_mllm{_rn50x4}.pth \
--img-path ./boost_ovd/metadata/coco_65_cls_emb_img{_rn50x4}.pth \
--pt-weight 1.0 --llm-weight 1.0 \
--mllm-weight 1.0 --img-weight 1.0 \
--dataset coco \
--save-path ./boost_ovd/metadata/coco_65_cls_emb_mean4{_rn50x4}.pth \
--base-save-path ./boost_ovd/metadata/coco_48_cls_emb_mean4{_rn50x4}.pth

# LVIS
python3 ./boost_ovd/CEE-MMHK/merge_embed.py \
--pt-path ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb{_rn50x4}.pth \
--llm-path ./boost_ovd/metadata/lvis_1203_cls_emb_llm{_rn50x4}.pth \
--mllm-path ./boost_ovd/metadata/lvis_1203_cls_emb_mllm{_rn50x4}.pth \
--img-path ./boost_ovd/metadata/lvis_1203_cls_emb_img{_rn50x4}.pth \
--pt-weight 1.0 --llm-weight 1.0 \
--mllm-weight 1.0 --img-weight 1.0 \
--dataset lvis \
--save-path ./boost_ovd/metadata/lvis_1203_cls_emb_mean4{_rn50x4}.pth \
--base-save-path ./boost_ovd/metadata/lvis_866_base_cls_emb_mean4{_rn50x4}.pth


