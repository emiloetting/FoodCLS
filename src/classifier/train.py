print("Importing..")
import json
from pathlib import Path
import sys
import duckdb as ddb
import pandas as pd
import torch
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parents[1]))
from classifier.preparation import DataProcessor, FoodDataset
from classifier.archs import MLP

CRITERION = torch.nn.CrossEntropyLoss
OPTIMIZER = torch.optim.Adam
BATCH_SIZE = 1024
MODEL = MLP
EPOCHS = 100
RUNS = 3

# Create dir to keep contents form training run
train_start_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
models_dir = Path(__file__).parents[2] / "models" / train_start_time
models_dir.mkdir(parents=True, exist_ok=True)

# Load config
with open(Path(__file__).parents[2]/"data"/"config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Reproducability
torch.manual_seed(int(config["seed"]))
torch.backends.cudnn.deterministic = True

# Log training specs
store_dir = {
    "optimizer": OPTIMIZER.__name__,
    "criterion": CRITERION.__name__,
    "batch": BATCH_SIZE,
    "seed": int(config["seed"]),
    "model": MLP.__name__,
    "epochs": EPOCHS,
    "runs": RUNS
}
with open(models_dir/"meta.json", "w", encoding="utf-8") as f:
    json.dump(store_dir, f, ensure_ascii=False)


# Collect data for training
db_parquet = Path(config["original_db"])  # extract path to db
tags_csv_pth = Path(config["tags_csv"])

# Load data dirty (all in RAM)
if not tags_csv_pth.is_file():
    res_df = ddb.sql(
        f"SELECT code, categories_tags FROM read_parquet('{db_parquet.resolve().as_posix()}')"
    ).df()

else:
    print("Loading data from csv...")
    res_df = pd.read_csv(
        tags_csv_pth,
        usecols=lambda column: column not in ["Unnamed: 0", "product_name"],    # ignore product name since not important for inference
        dtype={"code": "string"},
    )

# Init processor to ship with model
processor = DataProcessor(
   Path(config["orig_unique_categories_json"]).resolve()
)
processor.save_to_pkl(models_dir/"processor.pkl")

# Load dict with barcode-label mappings
print("Collecting tags...")
with open(config["codes_for_training_json"], "r", encoding="utf-8") as f:
    llm_mapping_dict = json.load(f)     # dirty-assume dict (I created the file..)
normalized_mapping = {
    str(code).lstrip("0") or "0": label
    for code, label in llm_mapping_dict.items()
}
normalized_codes = res_df["code"].astype("string").str.lstrip("0").replace("", "0")
mapped_labels = normalized_codes.map(normalized_mapping)
matched = mapped_labels.notna()
training_rows = res_df.loc[matched, ["categories_tags"]].copy()
training_rows["label"] = mapped_labels.loc[matched].astype("int64")
training_rows["categories_tags"] = training_rows["categories_tags"].apply(
    processor.normalize_tags
)
training_rows = training_rows[training_rows["categories_tags"].str.len() > 0]
to_embed = training_rows["categories_tags"].tolist()
labels = training_rows["label"].tolist()

print("To embed: ", len(to_embed))
print("Labels: ", len(labels))
print("Unique labels: ", set(labels))

# Create embeddings and build dataset
embeds = [processor.vectorize_tags(tag_list) for tag_list in to_embed]

# Build datasets and dataloader
grand_ds = FoodDataset(embeds, labels)
train_ds, rest_ds = grand_ds._data_split(0.7, 0.3)
val_ds, test_ds = rest_ds._data_split(0.67, 0.33)

train_loader = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, BATCH_SIZE, shuffle=True)