# Generate class-agnostic pseudo-labels

# COCO
python3 ./boost_ovd/LNE-CAPL/pseudo.py \
--gt-path ./datasets/coco/annotations/ovd_ins_train2017_b.json \
--save-path ./datasets/coco/annotations/ovd_ins_train2017_b_unclassified_pseudo.json \
--text-prompt object \
--box-threshold 0.1 \
--text-threshold 0.1 \
--is-coco True

# LVIS
python3 ./boost_ovd/LNE-CAPL/pseudo.py \
--gt-path ./datasets/lvis/lvis_v1_train_norare.json \
--save-path ./datasets/lvis/lvis_v1_train_norare_unclassified_pseudo.json \
--text-prompt object \
--box-threshold 0.1 \
--text-threshold 0.1 \
--is-coco False

# Offline train localization network

# COCO
python tools/train_net.py \
--num-gpus 4 \
--config-file ./configs/Localization_Network/localization_network_R_50_C4_1x_coco_ovd_pseudo.yaml

# LVIS (for zero-shot inference)
python tools/train_net.py \
--num-gpus 4 \
--config-file ./configs/Localization_Network/localization_network_R_50_FPN_1x_lvis_ovd_pseudo.yaml

# LVIS (for transfer learning)
python tools/lazyconfig_train_net.py \
--num-gpus 4 \
--config-file ./configs/Localization_Network/localization_network_R_50_FPN_400ep_LSJ_lvis_ovd_pseudo.py
