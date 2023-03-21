from translator import Translator
from datasets import load_dataset
import json
import jsonlines
from tqdm import tqdm
import re

# Following models are fpr Indonesian Machine Translation. Feel free to update it for your language
model_name = "Wikidepia/marian-nmt-enid"
# model_name = "Helsinki-NLP/opus-mt-en-id"
translator = Translator(model_name, cache_enabled=True)


def fix_number(source, target):
    numbers = re.findall(r"\n(\d{1,2}\.?)", source)
    for i, number in enumerate(numbers):
        target = re.sub(f"{number}", f"\n{number}", target)
    return target


def main():
    alpaca = load_dataset("tatsu-lab/alpaca")
    print("Data length:", len(alpaca["train"]))

    task_1 = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    task_2 = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    task_1_translated = translator.translate([task_1])[0]
    task_2_translated = translator.translate([task_2])[0]

    ds = []
    for i, row in tqdm(enumerate(alpaca["train"]), total=alpaca["train"].num_rows):
        comas = row["output"].split(",")
        if len(comas) >= 50:
            print(i, row["output"])
            continue
        if row["input"] == "":
            instruction = translator.translate([row["instruction"]])[0]
            response = fix_number(row["output"], translator.translate([row["output"]])[0])
            ds.append({
                "text": f'{task_2_translated}\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}',
            })
        else:
            instruction = translator.translate([row["instruction"]])[0]
            input_ = fix_number(row["input"], translator.translate([row["input"]])[0])
            response = fix_number(row["output"], translator.translate([row["output"]])[0])
            ds.append({
                "text": f'{task_1_translated}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:\n{response}',
            })

    with jsonlines.open(f'alpaca_id.jsonl', mode='w') as writer:
        writer.write_all(ds)


if __name__ == "__main__":
    main()
