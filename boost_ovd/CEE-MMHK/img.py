import argparse
import json
from collections import defaultdict
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import time
import openai
import requests
from openai.error import RateLimitError, InvalidRequestError

openai.api_key = "Your OpenAI API KEY"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann-path",
        type=str,
        default="./datasets/coco/annotations/ovd_ins_val2017_all.json"
        # "./datasets/coco/annotations/ovd_ins_train2017_all.json" or "./datasets/lvis/lvis_v1_train.json"
    )
    parser.add_argument(
        "--use-t2i",
        type=bool,
        default=False # False True
    )
    parser.add_argument(
        "--t2i-root",
        type=str,
        default=None  # None "./ours/metadata/lvis_ovd_t2i_imgs"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="dall-e-2"
    )
    parser.add_argument(
        "--img-size",
        type=str,
        default="512x512"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./ours/metadata/coco_ovd_image_examples.json"
        # "./ours/matadata/coco_ovd_image_examples.json" or "./ours/metadata/coco_ovd_image_examples.json"
    )
    parser.add_argument(
        "--base-num",
        type=int,
        default=5
    )
    parser.add_argument(
        "--novel-num",
        type=int,
        default=5
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=4096  # COCO:4096 LVIS:512
    )
    return parser

def download_img(urls,save_root):
    for i,url in enumerate(urls):
        try:
            f = requests.get(url)
            with open(os.path.join(save_root,"{}.jpg".format(i)), "wb") as code:
                code.write(f.content)
            print('{} success'.format(i))
        except:
            print('{} fail'.format(url))

def main(args):
    with open(args.ann_path, 'r') as f:
        gt = json.load(f)

    # Generate images using T2I
    if args.use_t2i:
        categories = sorted(gt['categories'], key=lambda x: x["id"])

        category_list = [] # [(list(cate_name),cate_def)]
        for c in categories:
            # LVIS: only 'c' anf 'r'
            if 'frequency' in c and c['frequency'] == 'f':
                continue
            if 'supercategory' in c:    # COCO
                category_list.append(([c['name']],c['supercategory']))
            else:                       # LVIS
                if c['name'] in c['synonyms']:
                    c['synonyms'].remove(c['name'])
                category_list.append(([c['name']] + c['synonyms'],c['def']))
        print('Cls Num:',len(category_list))

        vowel_list = ['a', 'e', 'i', 'o', 'u']

        failed_clss = []
        for i, (categorys,info) in enumerate(category_list):
            save_root = os.path.join(args.t2i_root,categorys[0].replace(' ','_'))
            if os.path.exists(save_root) and len(os.listdir(save_root)) == args.K:
                continue

            # make prompts
            prompts = []
            for i,category in enumerate(categorys):
                if category[0] in vowel_list:
                    article = 'an'
                else:
                    article = 'a'
                if i == 0:
                    prompts.append(f"A photo of {article} {category.replace('_',' ')}, which is {info}.")
                    prompts.append(f"A photo of {article} {category.replace('_', ' ')}.")
                    prompts.append(f"A photo of {info}.")
                else:
                    prompts.append(f"A photo of {article} {category.replace('_', ' ')}.")

            response = None
            for prompt in prompts:
                print('Prompt:',prompt)
                try:
                    response = openai.Image.create(
                        prompt=prompt,
                        model=args.openai_model,
                        quality="standard",
                        size=args.img_size,
                        response_format="url",
                        style='natural',
                        n=args.K
                    )
                    break
                except RateLimitError:
                    print("Hit rate limit. Waiting 15 seconds.")
                    time.sleep(15)
                    response = openai.Image.create(
                        prompt=prompt,
                        model=args.openai_model,
                        quality="standard",
                        size=args.img_size,
                        response_format="url",
                        style='natural',
                        n=args.K
                    )
                    break
                except InvalidRequestError:
                    if prompt == prompts[-1]:
                        print('prompt is sensitive!')
                        failed_clss.append(categorys)
                        break
                    time.sleep(5)
                    continue

            if not response:
                continue

            time.sleep(0.15)

            urls = [r['url'] for r in response["data"]]

            os.makedirs(save_root,exist_ok=True)
            download_img(urls,save_root)

        print('Done!')
        print(f'Failed Classes:{failed_clss}')

    # Choice Image Examples
    # transform to COCO format
    if 'file_name' not in gt['images'][0]:
        for img in gt['images']:
            img['file_name'] = img['coco_url'].replace('http://images.cocodataset.org/','')

    # get novel cat ids
    novel_cat_ids = []
    extra_need_nums = {}
    for cat in gt['categories']:
        if 'split' in cat:  # COCO
            if cat['split'] == 'unseen':
                novel_cat_ids.append(cat['id'])
                extra_need_nums[cat['id']] = args.novel_num
            else:
                extra_need_nums[cat['id']] = args.base_num
        elif 'frequency' in cat:  # LVIS
            if cat['frequency'] == 'r':
                novel_cat_ids.append(cat['id'])
                extra_need_nums[cat['id']] = args.novel_num
            else:
                extra_need_nums[cat['id']] = args.base_num
    print('Novel:', len(novel_cat_ids))

    # get sample annos
    catid2annos = defaultdict(list)
    for anno in gt['annotations']:
        # filter small object
        if anno['area'] <= args.min_size:
            continue
        # filter COCO's 'iscrowd'
        if 'iscrowd' in anno and anno['iscrowd'] == 1:
            continue
        catid2annos[anno['category_id']].append(anno)

    # random choice by area
    chosen_anns_all = []
    rng = np.random.default_rng(args.seed)
    for cat_id in tqdm(sorted(catid2annos.keys()), total=len(catid2annos)):
        annos = catid2annos[cat_id]
        probs = [a['area'] for a in annos]
        probs = np.array(probs) / sum(probs)
        nums = min(args.base_num, len(annos)) if cat_id not in novel_cat_ids else min(args.novel_num, len(annos))
        chosen_anns = rng.choice(annos, size=nums, p=probs, replace=False)
        chosen_anns_all.extend(chosen_anns.tolist())
        extra_need_nums[cat_id] -= nums

    # supplement extra data
    if args.use_t2i:
        extra_data_basename = os.path.basename(args.t2i_root)
        catid2name = {c['id']:c['name'] for c in gt['categories']}
        img_id, anno_id = max([img['id'] for img in gt['images']]), max([anno['id'] for anno in gt['annotations']])
        for cat_id, extra_num in extra_need_nums.items():
            if extra_num == 0:
                continue
            cat_name = catid2name[cat_id]
            cat_root = os.path.join(args.t2i_root, cat_name)
            if not os.path.exists(cat_root):
                continue

            for file_name in random.sample(os.listdir(cat_root),min(len(os.listdir(cat_root)),extra_num)):
                if not file_name.endswith('jpg'):
                    continue
                img_path = os.path.join(cat_root, file_name)
                with Image.open(img_path) as img:
                    width, height = img.size
                img_id += 1
                gt["images"].append({
                    'file_name': os.path.join(extra_data_basename, cat_name, file_name),  # COCO format
                    'width': width,
                    'height': height,
                    'id': img_id
                })
                anno_id += 1
                chosen_anns_all.append({
                    'id': anno_id,
                    'image_id': img_id,
                    'category_id': cat_id,
                    'area': width * height,
                    'bbox': [0, 0, width, height]
                })

    # filter images
    img_ids = set()
    for anno in chosen_anns_all:
        img_ids.add(anno['image_id'])
    remain_imgs = [img for img in gt['images'] if img['id'] in img_ids]

    # save
    sample_gt = {}
    sample_gt['categories'] = gt['categories']
    sample_gt['images'] = remain_imgs
    sample_gt['annotations'] = chosen_anns_all

    with open(args.output_path, 'w') as f:
        json.dump(sample_gt, f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)