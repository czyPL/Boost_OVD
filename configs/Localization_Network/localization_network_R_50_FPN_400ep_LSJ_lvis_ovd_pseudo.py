from ..new_baselines.mask_rcnn_R_50_FPN_400ep_LSJ import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.output_dir = ('./output/localization_network_fpn_400ep_lsj_lvis_ovd_pseudo')

model.roi_heads.num_classes = 1
model.roi_heads.mask_in_features = None # not use mask head

# stop RegionCLIP
model.use_clip_c4 = False
model.use_clip_attpool =  False
model.roi_heads.box_predictor.bg_cls_loss_weight = None
model.roi_heads.box_predictor.openset_test = (None,None,None,None)
model.roi_heads.box_predictor.clip_cls_emb_fuse = None

dataloader.train.dataset.names = "lvis_v1_norare_pseudo"
dataloader.train.mapper.use_instance_mask = False # not use mask head
dataloader.train.mapper.recompute_boxes = False
# not eval
del dataloader.test
del dataloader.evaluator