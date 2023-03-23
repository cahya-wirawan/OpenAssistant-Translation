from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
"""
languages = ['ar', 'bg', 'bn', 'ca', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'hi', 'ht',
             'id', 'it', 'ja', 'ko', 'ms', 'pt', 'ru', 'sw', 'ta', 'th', 'tr', 'ur', 'vi', 'zh']
"""
languages = ['bg', 'bn', 'ca', 'de', 'en', 'es', 'et', 'fi', 'fr', 'hi', 'ht',
             'id', 'it', 'ja', 'ms', 'pt', 'ru', 'sw', 'ta', 'th', 'tr', 'vi', 'zh']

def main():
    ds_all = None
    for i, lang in enumerate(languages):
        ds_name = f"cahya/instructions-{lang}"
        ds = load_dataset(ds_name)
        ds = ds.remove_columns(["id"])
        print(f"{lang}: {len(ds['train'])}, {len(ds['validation'])}, {len(ds['test'])}")
        if i == 0:
            ds_all = ds
        else:
            for split in ["train", "validation", "test"]:
                ds_all[split] = concatenate_datasets([ds_all[split], ds[split]])
    print(f"Dataset all: {len(ds_all['train'])}, {len(ds_all['validation'])}, {len(ds_all['test'])}")
    ds_all = ds_all.shuffle(seed=42)
    ds_all.push_to_hub("cahya/instructions-all")


if __name__ == "__main__":
    main()