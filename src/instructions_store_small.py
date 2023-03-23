import json
from datasets import Dataset


# languages = ['de', 'en', 'es', 'fr', 'hi', 'id', 'ja', 'ms', 'pt', 'ru', 'th', 'vi', 'zh']
languages = ['ar', 'bg', 'bn', 'ca', 'el', 'et', 'fi', 'ht', 'it', 'ko', 'sw', 'ta', 'tr', 'ur']


def main():
    for lang in languages:
        instruction_name = f"instruction_{lang}.jsonl"
        with open(instruction_name, "r") as file:
            data = []
            for i, line in enumerate(file):
                row = json.loads(line)
                data.append({"id": i, "text": row["label"][0]})
        print(f"Data length of {lang}: {len(data)}")

        ds = Dataset.from_list(data).train_test_split(test_size=0.1, shuffle=True, seed=42)
        test_validation = ds["test"].train_test_split(test_size=0.5, shuffle=True, seed=42)
        ds["validation"] = test_validation["train"]
        ds["test"] = test_validation["test"]
        # print(ds)
        # print(ds["train"][0])
        ds.push_to_hub(f"cahya/instructions-{lang}")


if __name__ == "__main__":
    main()