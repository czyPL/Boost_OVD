from .mask_rcnn_regnety_4gf_dds_FPN_100ep_LSJ import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter *= 2  # 100ep -> 200ep
