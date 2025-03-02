# Prepare Pretrained Models

We start with [RegionCLIP](https://github.com/microsoft/RegionCLIP/tree/main), so we need to download its pretrained model first.

## Model downloading:
The RegionCLIP pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1hzrJBvcCrahoRcqJRqzkIGFO_HUSJIii?usp=sharing). What we need includes:

- `regionclip`: Pretrained RegionCLIP models.
- `concept_emb`: The embeddings for object class names.

You need to download the following files and put them into respective folder in `pretrained_ckpt/`. The file structure should be
```
pretrained_ckpt/
  regionclip/
    regionclip_pretrained-cc_rn50.pth
    regionclip_pretrained-cc_rn50x4.pth
  concept_emb/
    coco_65_cls_emb.pth
    coco_65_cls_emb_rn50x4.pth
    lvis_1203_cls_emb.pth
    lvis_1203_cls_emb_rn50x4.pth 
```

The following is a description of these files.

- `regionclip_pretrained-cc_{rn50, rn50x4}.pth`: Using Google Conceptual Caption (3M image-text pairs) to pretrain RegionCLIP. `rn50` denotes that both teacher and student visual backbones are ResNet50. `rn50x4` represents ResNet50x4.

- `{coco, lvis}_NUMBER_emb*.pth`: These files store the class embeddings of COCO and LVIS datasets. Computing the embeddings of object names using CLIP's language encoder and exporting them as the following local files. By default, the embeddings are obtained using CLIP with ResNet50 architecture, unless noted otherwise (eg, `rn50x4` denotes ResNet50x4). `NUMBER` denotes the number of classes.