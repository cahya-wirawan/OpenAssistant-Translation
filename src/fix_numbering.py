import jsonlines
import json
import re

# It fixes the numbering/newline issues after translation.
# Since this is for the result of Indonesian translation "instructions_id.jsonl",
# you might need to change the word "Asisten" accordingly

with open("instructions_id.jsonl", "r") as file:
    with jsonlines.open(f'instructions_id_fixed.jsonl', mode='w') as writer:
        for line in file:
            row = json.loads(line)
            numbers = re.findall(r"\n(\d{1,2}\.)", row["text"])
            row["label"][0] = re.sub(f" Asisten:", f"\nAsisten:", row["label"][0])
            for number in numbers:
                row["label"][0] = re.sub(f" {number}", f"\n{number}", row["label"][0])
            writer.write(row)