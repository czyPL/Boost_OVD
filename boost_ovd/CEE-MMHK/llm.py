import argparse
import time
import json
import openai
from openai.error import RateLimitError

openai.api_key = "Your OpenAI API KEY"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann-path",
        type=str,
        default="./datasets/coco/annotations/ovd_ins_val2017_all.json"
        # "./datasets/coco/annotations/ovd_ins_val2017_all.json" "./datasets/lvis/lvis_v1_val.json"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./ours/metadata/coco_ovd_llm_descriptions.json"
        # "./ours/metadata/coco_ovd_llm_descriptions.json" "./ours/metadata/lvis_ovd_llm_descriptions.json"
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-3.5-turbo-instruct"
    )
    return parser


def main(args):
    with open(args.ann_path, 'r') as f:
        gt = json.load(f)

    categories = sorted(gt['categories'], key=lambda x: x["id"])

    category_list = []
    for c in categories:
        if 'supercategory' in c:
            category_list.append((c['name'],c['supercategory']))
        else:
            category_list.append((c['name'],c['def']))

    all_responses = {}
    vowel_list = ['a', 'e', 'i', 'o', 'u']

    for i, (category,info) in enumerate(category_list):
        if category[0] in vowel_list:
            article = 'an'
        else:
            article = 'a'
        prompts = []
        prompt = f"Describe what {article} {category} usually looks like, {category} is {info} thing."
        prompts.append(prompt)

        all_result = []
        # call openai api taking into account rate limits
        for curr_prompt in prompts:
            try:
                response = openai.Completion.create(
                    model=args.openai_model,
                    prompt=curr_prompt,
                    temperature=0.99,
                    max_tokens=50,
                    n=100,
                    stop="."
                )
            except RateLimitError:
                print("Hit rate limit. Waiting 15 seconds.")
                time.sleep(15)
                response = openai.Completion.create(
                    model=args.openai_model,
                    prompt=curr_prompt,
                    temperature=.99,
                    max_tokens=50,
                    n=100,
                    stop="."
                )

            time.sleep(0.15)

            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                all_result.append(result.replace("\n", "").strip() + ".")
        all_responses[category] = all_result

        # info
        print(prompt)
        for i in all_result:
            print(i)

    with open(args.output_path, 'w') as f:
        json.dump(all_responses, f, indent=4)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)