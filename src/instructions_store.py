import json
from datasets import Dataset


instruction_name = "instruction_id.jsonl"


def main():
    with open(instruction_name, "r") as file:
        data = []
        for i, line in enumerate(file):
            row = json.loads(line)
            data.append({"id": i, "text": row["label"][0].replace("Coninue", "Teruskan")})
    print("Data length:", len(data))

    ds = Dataset.from_list(data).train_test_split(test_size=0.1, shuffle=True, seed=42)
    test_validation = ds["test"].train_test_split(test_size=0.5, shuffle=True, seed=42)
    ds["validation"] = test_validation["train"]
    ds["test"] = test_validation["test"]
    print(ds)
    ds.push_to_hub("cahya/instructions-id")


if __name__ == "__main__":
    main()