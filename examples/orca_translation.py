from datasets import load_dataset
from open_translation import OpenTranslation
import jsonlines
from tqdm import tqdm
import re
import argparse
import sys

# Following models are for Indonesian Machine Translation. Feel free to update it for your language
model_name = "Wikidepia/marian-nmt-enid"


# model_name = "Helsinki-NLP/opus-mt-en-id"


def fix_number(source, target):
    numbers = re.findall(r"\n(\d{1,2}\.?)", source)
    for i, number in enumerate(numbers):
        target = re.sub(f"{number}", f"\n{number}", target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--max_row", type=int, required=False, default=sys.maxsize,
                        help="The maximal row to be translated")
    parser.add_argument("-s", "--start", type=int, required=False, default=0,
                        help="The start index")
    parser.add_argument("-e", "--end", type=int, required=False, default=0,
                        help="The end index")
    parser.add_argument("-c", "--cache_name", type=str, required=False, default="default_cache",
                        help="The cache name")
    parser.add_argument("-o", "--output", type=str, required=False, default="orca_id.jsonl",
                        help="The output file")
    parser.add_argument("-d", "--debug", action='store_true', required=False, default=False,
                        help="Debug mode")
    args = parser.parse_args()

    translator = OpenTranslation(model_name, cache_enabled=True, cache_name=args.cache_name)
    if args.debug:
        dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        dataset_short = []
        for i, row in tqdm(enumerate(dataset)):
            if i >= 20:
                break
            dataset_short.append(row)
        dataset = dataset_short
    else:
        dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=False)
    args.end = len(dataset) if args.end == 0 else args.end
    with jsonlines.open(args.output, "w") as jsonl:
        # for i, row in tqdm(enumerate(dataset["train"]), total=len(dataset["train"])):
        for i in tqdm(range(args.start, args.end), total=args.end - args.start):
            if i >= args.max_row:
                break
            row = dataset[i]
            system_prompt_ = row["system_prompt"].split("\n")
            question_ = row["question"].split("\n")
            response_ = row["response"].split("\n")
            source = system_prompt_ + question_ + response_
            try:
                target = translator.translate(source)
            except IndexError as error:
                print(i, error)
                continue
            for j, text in enumerate(source):
                found = re.search(r"^(\s+)", text)
                if found:
                    target[j] = found.group(0) + target[j]
            system_prompt = "\n".join(target[:len(system_prompt_)])
            question = "\n".join(target[len(system_prompt_):len(system_prompt_) + len(question_)])
            response = "\n".join(target[len(system_prompt_) + len(question_):])
            soda_translation = {
                "id": row['id'],
                "system_prompt": system_prompt,
                "question": question,
                "response": response
            }
            jsonl.write(soda_translation)


if __name__ == "__main__":
    main()
