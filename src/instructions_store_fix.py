import json
import jsonlines


instruction_name = "instruction_id.jsonl"
instruction_name_fix = "instruction_id_fix.jsonl"


def main():
    with open(instruction_name, "r") as file:
        with jsonlines.open(instruction_name_fix, mode="w") as writer:
            for i, line in enumerate(file):
                row = json.loads(line)
                row["text"] = row["text"].replace("Coninue", "Continue")
                row["label"][0] = row["label"][0].replace("Coninue", "Teruskan")
                writer.write(row)


if __name__ == "__main__":
    main()