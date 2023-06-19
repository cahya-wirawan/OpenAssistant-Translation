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
    parser.add_argument("-o", "--output", type=str, required=False, default="soda_id.jsonl",
                        help="The output file")
    args = parser.parse_args()

    translator = OpenTranslation(model_name, cache_enabled=True, cache_name=args.cache_name)
    dataset = load_dataset("allenai/soda")
    soda_keys = ["head", "tail", "literal", "narrative", "dialogue"]
    soda_translation = []
    with jsonlines.open(args.output, "w") as jsonl:
        # for i, row in tqdm(enumerate(dataset["train"]), total=len(dataset["train"])):
        for i in tqdm(range(args.start, args.end), total=args.end-args.start):
            if i >= args.max_row:
                break
            row = dataset["train"][i]
            source = []
            for sk in soda_keys:
                if sk == "dialogue":
                    source += row[sk]
                else:
                    source.append(row[sk])
            target = translator.translate(source)
            translation = {}
            for j, sk in enumerate(soda_keys):
                if sk == "dialogue":
                    translation[sk] = target[j:]
                else:
                    translation[sk] = target[j]
            for key in dataset["train"].features:
                if key not in soda_keys:
                    translation[key] = row[key]
            soda_translation.append(translation)
            jsonl.write(translation)


if __name__ == "__main__":
    main()
