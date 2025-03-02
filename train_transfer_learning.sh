# train open-vocabulary object detectors, {RN50, RN50x4} x {COCO, LVIS}
# initialized by pretrained RegionCLIP and using our enhanced localization network and class embedding)

# RN50, COCO
python3 ./tools/train_net.py \
--num-gpus 4 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./output/localization_network_c4_coco_ovd_pseudo/model_final.pth \
MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/coco_48_base_cls_emb_mean4.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./ours/metadata/coco_65_cls_emb_mean4.pth \
OUTPUT_DIR ./output/coco_rn50

 # RN50, LVIS
 python3 ./tools/train_net.py \
 --num-gpus 4 \
 --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
 MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
 MODEL.CLIP.BB_RPN_WEIGHTS ./output/localization_network_fpn_400ep_lsj_lvis_ovd_pseudo/model_final.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/lvis_866_base_cls_emb_mean4.pth \
 MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./ours/metadata/lvis_1203_cls_emb_mean4.pth \
 MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
 OUTPUT_DIR ./output/lvis_rn50

 # RN50x4, COCO
 python3 ./tools/train_net.py \
 --num-gpus 4 \
 --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
 MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
 MODEL.CLIP.BB_RPN_WEIGHTS ./output/localization_network_c4_coco_ovd_pseudo/model_final.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/coco_48_base_cls_emb_mean4_rn50x4.pth \
 MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./ours/metadata/coco_65_cls_emb_mean4_rn50x4.pth \
 MODEL.CLIP.TEXT_EMB_DIM 640 \
 MODEL.RESNETS.DEPTH 200 \
 MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
 OUTPUT_DIR ./output/coco_rn50x4

 # RN50x4, LVIS
 python3 ./tools/train_net.py \
 --num-gpus 1 \
 --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
 MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
 MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
 MODEL.CLIP.BB_RPN_WEIGHTS ./output/localization_network_fpn_400ep_lsj_lvis_ovd_pseudo/model_final.pth \
 MODEL.CLIP.TEXT_EMB_PATH ./ours/metadata/lvis_866_base_cls_emb_mean4_rn50x4.pth \
 MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./ours/metadata/lvis_1203_cls_emb_mean4_rn50x4.pth \
 MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
 MODEL.CLIP.TEXT_EMB_DIM 640 \
 MODEL.RESNETS.DEPTH 200 \
 MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
 MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION 18 \
 MODEL.RESNETS.RES2_OUT_CHANNELS 320 \
 OUTPUT_DIR ./output/lvis_rn50x4