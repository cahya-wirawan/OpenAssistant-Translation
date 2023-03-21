from translator import Translator
import json
import jsonlines
from tqdm import tqdm

# Following models are fpr Indonesian Machine Translation. Feel free to update it for your language
model_name = "Wikidepia/marian-nmt-enid"
#model_name = "Helsinki-NLP/opus-mt-en-id"
translator = Translator(model_name, cache_enabled=True)

with open("oa_v3_fixed_plus_safety.jsonl", "r") as file:
    data = [json.loads(line) for line in file]
print("Data length:", len(data))
ds = []
for i, row in tqdm(enumerate(data), total=len(data)):
    if row["meta"]["source"] not in ["synth_code", "conala"]:
        ds.append(
            {
                "text": row["text"],
                "label": [translator.translate([row["text"]])[0]]
            }
        )
with jsonlines.open(f'instruction_id.jsonl', mode='w') as writer:
     writer.write_all(ds)
