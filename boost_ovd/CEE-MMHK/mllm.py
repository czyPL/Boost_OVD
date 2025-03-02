import argparse
import os
import json
import torch
from concurrent import futures
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann-path",
        type=str,
        default="./datasets/coco/annotations/ovd_ins_train2017_all.json",
        # "./datasets/coco/annotations/ovd_ins_train2017_all.json" or "./datasets/lvis/lvis_v1_train.json"
        help='path of the annotation file'
    )
    parser.add_argument(
        "--img-root",
        type=str,
        default="datasets/coco",
        help='path of the image'
    )
    parser.add_argument(
        "--crop_path",
        type=str,
        default="./datasets/crop/coco_crop_150_5",
        # "./datasets/crop/coco_crop_150_5" or "./datasets/crop/lvis_crop_150_5"
        help='path to save the object crops'
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./ours/metadata/coco_ovd_mllm_descriptions.json",
        # "./ours/metadata/coco_ovd_mllm_descriptions.json" or "./ours/metadata/lvis_ovd_mllm_descriptions.json"
        help='path to save the all result'
    )
    parser.add_argument(
        "--base-crop-num",
        type=int,
        default=150,
        help='crop number per base category'
    )
    parser.add_argument(
        "--novel-crop-num",
        type=int,
        default=5,
        help='crop number per novel category'
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=0,
        help='factor for extending the object box'
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=1024.0,
        help='the min size of the box'
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=10,
        help='the number of the thread to save the images'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407
    )
    parser.add_argument(
        "--extra-data",
        type=str,
        default=None
        # None "./our/metadata/lvis_ovd_t2i_imgs"
    )
    return parser

def gen_data(file_dir):
    file_paths = []
    for root, dirs, files in os.walk(file_dir):
        for filename in files:
            if os.path.splitext(filename)[1] in ['.jpg', '.png', '.jpeg']:
                path = os.path.join(root, filename)
                file_paths.append(path)
    return file_paths

def cropping_fun(image_anns, img_root, save_root, catid2name, imgid2name, scaling_factor):
    for i in tqdm(range(len(image_anns)), total=len(image_anns)):
        image_ann = image_anns[i]
        cat_id = image_ann['category_id']
        img_id = image_ann['image_id']

        cat_name = catid2name[cat_id]
        img_name = imgid2name[img_id]
        img_path = os.path.join(img_root, img_name)
        save_cat_root = os.path.join(save_root, cat_name)
        os.makedirs(save_cat_root, exist_ok=True)
        save_path = os.path.join(save_cat_root, str(i) + '_' + img_name.split('/')[-1])

        x, y, w, h = image_ann['bbox']

        try:
            image = Image.open(img_path)
            xmin = max(0, int(x - scaling_factor * w))
            ymin = max(0, int(y - scaling_factor * h))
            xmax = min(image.size[0], int(x + (1 + scaling_factor) * w))
            ymax = min(image.size[1], int(y + (1 + scaling_factor) * h))
            image = image.crop([xmin, ymin, xmax, ymax])

            image.save(save_path)
        except:
            print(f"{save_path} error!")

    return True


def main(args):
    # STEP 1： Generate the object crops from GT
    with open(args.ann_path, 'r') as f:
        gt = json.load(f)

    # get catid2name and base/novel id
    catid2name = {}
    base_id, novel_id = [], []
    for cat in gt['categories']:
        catid2name[cat['id']] = cat['name']
        if 'split' in cat:  # COCO
            if cat['split'] == 'seen':
                base_id.append(cat['id'])
            else:
                novel_id.append(cat['id'])
        elif 'frequency' in cat:  # LVIS
            if cat['frequency'] == 'r':
                novel_id.append(cat['id'])
            else:
                base_id.append(cat['id'])
    print('Base:', len(base_id), 'Novel:', len(novel_id))
    print('Origin Annotations:', len(gt['annotations']))

    # combine extra data
    if args.extra_data is not None:
        extra_data_basename = os.path.basename(args.extra_data)
        # Images and Annotations
        images, annotations = [], []
        img_id, anno_id = max([img['id'] for img in gt['images']]), max([anno['id'] for anno in gt['annotations']])
        for catid, catname in catid2name.items():
            cat_root = os.path.join(args.extra_data, catname)
            if not os.path.exists(cat_root):
                continue

            for file_name in os.listdir(cat_root):
                if not file_name.endswith('jpg'):
                    continue
                img_path = os.path.join(cat_root, file_name)
                with Image.open(img_path) as img:
                    width, height = img.size
                img_id += 1
                images.append({
                    'file_name': os.path.join(extra_data_basename, catname, file_name),  # COCO format
                    'width': width,
                    'height': height,
                    'id': img_id
                })
                anno_id += 1
                annotations.append({
                    'id': anno_id,
                    'image_id': img_id,
                    'category_id': catid,
                    'area': width * height,
                    'bbox': [0, 0, width, height]
                })
        gt['images'].extend(images)
        gt['annotations'].extend(annotations)
        print('New Annotations:', len(gt['annotations']))

    # get info
    imgid2name = {}
    for img in gt['images']:
        if 'coco_url' in img:
            phase, img_name = img['coco_url'].split('/')[-2:]
            imgid2name[img['id']] = os.path.join(phase, img_name)
        else:
            imgid2name[img['id']] = img['file_name']

    catid2annos = defaultdict(list)
    for anno in gt['annotations']:
        # filter small object
        if anno['area'] <= args.min_size:
            continue
        # filter COCO's 'iscrowd'
        if 'iscrowd' in anno and anno['iscrowd'] == 1:
            continue
        catid2annos[anno['category_id']].append(anno)

    for catid, annos in catid2annos.items():
        if catid in novel_id and len(annos) < args.novel_crop_num:
            print(f"Category {catid} don't have {args.novel_crop_num} objects, You need to use extra-data")
        if catid in base_id and len(annos) < args.base_crop_num:
            print(f"Category {catid} don't have {args.base_crop_num} objects, You need to use extra-data")

    # random choice (crop num) by area
    chosen_anns_all = []
    rng = np.random.default_rng(args.seed)
    for cat_id, annos in catid2annos.items():
        probs = [a['area'] for a in annos]
        probs = np.array(probs) / sum(probs)
        crop_num = min(args.base_crop_num, len(annos)) if cat_id in base_id else args.novel_crop_num
        chosen_anns = rng.choice(annos, size=crop_num, p=probs, replace=False)
        chosen_anns_all.extend(chosen_anns.tolist())

    if not os.path.exists(args.crop_path):
        os.mkdir(args.crop_path)

    count_per_thread = (len(chosen_anns_all) + args.num_thread - 1) // args.num_thread
    image_anns = [chosen_anns_all[i * count_per_thread:(i + 1) * count_per_thread] for i in range(args.num_thread)]

    with futures.ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        threads = [executor.submit(cropping_fun, image_ann, img_root=args.img_root, save_root=args.crop_path, \
                                   catid2name=catid2name, imgid2name=imgid2name, scaling_factor=args.scaling_factor) \
                   for image_ann in image_anns]
        for future in futures.as_completed(threads):
            print(future.result())

    # STEP 2： Generate MLLM Category Descriptions
    if not os.path.exists(args.output_path):
        # Load model weights
        processor = Blip2Processor.from_pretrained('blip2-opt-2.7b')
        model = Blip2ForConditionalGeneration.from_pretrained('blip2-opt-2.7b', device_map={"": 0})  # load_in_8bit=True,torch_dtype=torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        results = defaultdict(list)
        img_paths = gen_data(args.crop_path)
        for img_path in tqdm(img_paths):
            cls_name = os.path.basename(os.path.dirname(img_path))
            prompt = cls_name.replace('_', ' ')
            image = Image.open(img_path).convert('RGB')

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device)  # dtype=torch.float16

            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            generated_text = f'{prompt} {generated_text}.'

            results[cls_name].append(generated_text)

        # decouple
        for cls_name, texts in results.items():
            results[cls_name] = list(set(texts))

        # save
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w') as f:
            f.write(json.dumps(results, indent=4))
    else:
        print(f'{args.output_path} is already exists!')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
