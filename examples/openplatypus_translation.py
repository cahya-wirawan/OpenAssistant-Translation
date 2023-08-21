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
    parser.add_argument("-o", "--output", type=str, required=False, default="openplatypus_id.jsonl",
                        help="The output file")
    parser.add_argument("-d", "--debug", action='store_true', required=False, default=False,
                        help="Debug mode")
    args = parser.parse_args()

    translator = OpenTranslation(model_name, cache_enabled=True, cache_name=args.cache_name)
    dataset_name = "garage-bAInd/Open-Platypus"
    if args.debug:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        dataset_short = []
        for i, row in tqdm(enumerate(dataset)):
            if i >= 20:
                break
            dataset_short.append(row)
        dataset = dataset_short
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=False)
    args.end = len(dataset) if args.end == 0 else args.end
    with jsonlines.open(args.output, "w") as jsonl:
        # for i, row in tqdm(enumerate(dataset["train"]), total=len(dataset["train"])):
        for i in tqdm(range(args.start, args.end), total=args.end - args.start):
            if i >= args.max_row:
                break
            row = dataset[i]
            input_ = row["input"].split("\n")
            output_ = row["output"].split("\n")
            instruction_ = row["instruction"].split("\n")
            source = input_ + output_ + instruction_
            try:
                target = translator.translate(source)
            except IndexError as error:
                print(i, error)
                continue
            except KeyError as error:
                print(i, error)
                continue
            for j, text in enumerate(source):
                found = re.search(r"^(\s+)", text)
                if found:
                    target[j] = found.group(0) + target[j]
            input = "\n".join(target[:len(input_)])
            output = "\n".join(target[len(input_):len(input_) + len(output_)])
            instruction = "\n".join(target[len(input_) + len(output_):])
            openplatypus_translation = {
                "input": input,
                "output": output,
                "instruction": instruction
            }
            jsonl.write(openplatypus_translation)


if __name__ == "__main__":
    main()
