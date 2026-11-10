print("Importing..")
import json
from pathlib import Path
import sys
import duckdb as ddb
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm 
from sklearn.metrics import f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, str(Path(__file__).parents[1]))
from classifier.preparation import DataProcessor, FoodDataset
from classifier.archs import MLP

CRITERION = torch.nn.CrossEntropyLoss
OPTIMIZER = torch.optim.Adam
BATCH_SIZE = 2**15
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
    "runs": RUNS,
    "device": DEVICE
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
print()

# Create embeddings and build dataset
embeds = [processor.vectorize_tags(tag_list) for tag_list in to_embed]

# Build datasets and dataloader
grand_ds = FoodDataset(embeds, labels)
train_ds, rest_ds = grand_ds._data_split(0.7, 0.3)
val_ds, test_ds = rest_ds._data_split(0.67, 0.33)

train_loader = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, BATCH_SIZE, shuffle=True)

del embeds
del res_df
del llm_mapping_dict

run_tqdm = {
    "iterable": range(RUNS),
    "total": RUNS,
    "colour": "#e73b3b",
    "desc": "Run"
}

epoch_tqdm = {
    "iterable": range(EPOCHS),
    "total": EPOCHS,
    "colour": "#1856ff",
    "desc": "Epoch"
}

with open(config["target_classes"], "r", encoding="utf-8") as f:
    target_class_mapping = json.load(f)

amnt_classes = len(target_class_mapping.keys())

# Actual training
for run in tqdm(**run_tqdm):
    this_dir = models_dir / str(run)
    this_dir.mkdir(parents=True, exist_ok=True)

    model = MODEL(
        in_dims=processor.nr_of_known_tags,
        dropout_perc=0.2,
        classes=amnt_classes,
        label_mapping={}
    )
    model.to(DEVICE)

    criterion = CRITERION()
    optimizer = OPTIMIZER(model.parameters(), lr=0.001)

    # perform inference run before training
    with torch.no_grad():
        model.eval()

        correct = 0
        total = 0

        for vectors, labels in tqdm(train_loader, "batches", len(train_loader), False):
            vectors = vectors.to(DEVICE).float()
            labels = labels.to(DEVICE).long()

            outputs = model(vectors)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        tqdm.write(f"Run {run} | Initial Accuracy: {correct / total:.4f} | Initial F1: {f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro", zero_division=0):.4f}")

    best_val_loss = float("inf")    # to always store best model
    history = []

    for epoch in tqdm(**epoch_tqdm):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for vectors, labels in tqdm(train_loader, "Train Batches", len(train_loader), False):
            vectors = vectors.to(DEVICE).float()
            labels = labels.to(DEVICE).long()

            optimizer.zero_grad()

            outputs = model(vectors)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_preds = []
        val_targets = []
        with torch.no_grad():
            for vectors, labels in tqdm(val_loader, "Val Batches", len(val_loader), False):

                vectors = vectors.to(DEVICE).float()
                labels = labels.to(DEVICE).long()

                outputs = model(vectors)
                loss = criterion(outputs, labels)

                val_preds.extend(outputs.argmax(dim=1).cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        # store model  
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_loss / val_total,
            "val_accuracy": val_correct / val_total,
            "val_macro_f1": f1_score(
                    val_targets, val_preds, average="macro", zero_division=0
                )
        }

        history.append(metrics)

        tqdm.write(
            f"Run {run} | Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Train Acc: {metrics['train_accuracy']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Val Acc: {metrics['val_accuracy']:.4f} |"
            f"Val Macro F1: {metrics['val_macro_f1']:.4f}"
            ""
        )

        # Save best model
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            torch.save(model.state_dict(), this_dir / "best_model.pt")

    # Save training history
    with open(this_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    # Evaluate best model on test set
    model.load_state_dict(
        torch.load(this_dir / "best_model.pt", map_location=DEVICE, weights_only=True)
    )
    model.eval()

    test_correct = 0
    test_total = 0
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for vectors, labels in tqdm(test_loader, "Test Batches", len(test_loader), False):
            vectors = vectors.to(DEVICE).float()
            labels = labels.to(DEVICE).long()

            outputs = model(vectors)

            test_correct += (outputs.argmax(dim=1) == labels).sum().item()
            test_total += labels.size(0)

            test_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            test_targets.extend(labels.cpu().tolist())

            test_macro_f1 = f1_score(
                test_targets, test_preds, average="macro", zero_division=0
            )

print(f"Run {run} | Test Macro F1: {test_macro_f1:.4f}")
print(f"Run {run} | Test Accuracy: {test_correct / test_total:.4f}")