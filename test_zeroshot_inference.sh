# evaluate zero-shot inference {RN50, RN50x4} x {COCO, LVIS} x {GT, RPN}

# RN50, GT, COCO
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/coco_65_cls_emb_mean4.pth \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.CLIP.MULTIPLY_RPN_SCORE False

 # RN50, RPN, COCO
 python3 ./tools/train_net.py \
 --eval-only  \
 --num-gpus 1 \
 --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/coco_65_cls_emb_mean4.pth \
 MODEL.CLIP.CROP_REGION_TYPE RPN \
 MODEL.CLIP.MULTIPLY_RPN_SCORE True \
 MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml \
 MODEL.CLIP.BB_RPN_WEIGHTS ./output/localization_network_c4_coco_ovd_pseudo/model_final.pth

 # RN50, GT, LVIS
 python3 ./tools/train_net.py \
 --eval-only  \
 --num-gpus 1 \
 --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/lvis_1203_cls_emb_mean4.pth \
 MODEL.CLIP.CROP_REGION_TYPE GT \
 MODEL.CLIP.MULTIPLY_RPN_SCORE False \
 MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.0001

 # RN50, RPN, LVIS
 python3 ./tools/train_net.py \
 --eval-only  \
 --num-gpus 1 \
 --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/lvis_1203_cls_emb_mean4.pth \
 MODEL.CLIP.CROP_REGION_TYPE RPN \
 MODEL.CLIP.MULTIPLY_RPN_SCORE True \
 MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
 MODEL.CLIP.BB_RPN_WEIGHTS ./output/localization_network_fpn_lvis_ovd_pseudo/model_final.pth

 # RN50x4, GT, COCO
 python3 ./tools/train_net.py \
 --eval-only  \
 --num-gpus 1 \
 --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/coco_65_cls_emb_mean4_rn50x4.pth \
 MODEL.CLIP.CROP_REGION_TYPE GT \
 MODEL.CLIP.MULTIPLY_RPN_SCORE False \
 MODEL.CLIP.TEXT_EMB_DIM 640 \
 MODEL.RESNETS.DEPTH 200 \
 MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18

 # RN50x4, RPN, COCO
 python3 ./tools/train_net.py \
 --eval-only  \
 --num-gpus 1 \
 --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/coco_65_cls_emb_mean4_rn50x4.pth \
 MODEL.CLIP.CROP_REGION_TYPE RPN \
 MODEL.CLIP.MULTIPLY_RPN_SCORE True \
 MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml \
 MODEL.CLIP.BB_RPN_WEIGHTS ./output/localization_network_c4_coco_ovd_pseudo/model_final.pth \
 MODEL.CLIP.TEXT_EMB_DIM 640 \
 MODEL.RESNETS.DEPTH 200 \
 MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18


 # RN50x4, GT, LVIS
 python3 ./tools/train_net.py \
 --eval-only  \
 --num-gpus 1 \
 --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/lvis_1203_cls_emb_mean4_rn50x4.pth \
 MODEL.CLIP.CROP_REGION_TYPE GT \
 MODEL.CLIP.MULTIPLY_RPN_SCORE False \
 MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.0001 \
 MODEL.CLIP.TEXT_EMB_DIM 640 \
 MODEL.RESNETS.DEPTH 200 \
 MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18

 # RN50x4, RPN, LVIS
 python3 ./tools/train_net.py \
 --eval-only  \
 --num-gpus 1 \
 --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/lvis_1203_cls_emb_mean4_rn50x4.pth \
 MODEL.CLIP.CROP_REGION_TYPE RPN \
 MODEL.CLIP.MULTIPLY_RPN_SCORE True \
 MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
 MODEL.CLIP.BB_RPN_WEIGHTS ./output/localization_network_fpn_lvis_ovd_pseudo/model_final.pth \
 MODEL.CLIP.TEXT_EMB_DIM 640 \
 MODEL.RESNETS.DEPTH 200 \
 MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18