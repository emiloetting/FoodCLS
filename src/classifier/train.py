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
from classifier.classifiers import MLP

CRITERION = torch.nn.CrossEntropyLoss
OPTIMIZER = torch.optim.Adam
BATCH_SIZE = 4096
MODEL = MLP
EPOCHS = 50
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

# Load data dirty (all in RAM)
print("Loading from DB into DataFrame...")
res_df = ddb.sql(
    f"SELECT code, categories_tags FROM read_parquet('{db_parquet.resolve().as_posix()}')"
).df()


# Init processor to ship with model
processor = DataProcessor()
processor.load_vocab(config["vocab_json"])
processor.save(models_dir/"processor.pkl")


# Load dict with barcode-label mappings
print("Collecting tags...")
with open(config["barcode_mapping"], "r", encoding="utf-8") as f:
    barcode_mapping = json.load(f)

# Assign labels by barcode, preserving leading zeros.
training_rows = res_df[["categories_tags"]].copy()
training_rows["label"] = res_df["code"].astype("string").str.strip().map(barcode_mapping)
training_rows = training_rows.dropna(subset=["label"])
training_rows["label"] = training_rows["label"].astype("int64")
print(f"Label coverage: {len(training_rows):,} / {len(res_df):,} product rows")

# Keep only products with tags recognized by the processor.
training_rows["categories_tags"] = training_rows["categories_tags"].apply(
    processor.normalize_tags
)
training_rows = training_rows[training_rows["categories_tags"].str.len() > 0]
if training_rows.empty:
    raise ValueError("No products with both a mapped label and known tags remain.")
to_embed = training_rows["categories_tags"].tolist()
labels = training_rows["label"].tolist()

print("To embed: ", len(to_embed))
print("Labels: ", len(labels))
print("Unique labels: ", set(labels))
print()

# Build datasets and dataloader
grand_ds = FoodDataset(to_embed, labels, processor)
train_ds, rest_ds = grand_ds._data_split(0.7, 0.3, random_state=int(config["seed"]))
val_ds, test_ds = rest_ds._data_split(0.67, 0.33, random_state=int(config["seed"]))

train_loader = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, BATCH_SIZE, shuffle=True)

del to_embed
del grand_ds
del rest_ds
del res_df
del barcode_mapping
del training_rows

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


# Actual training
for run in tqdm(**run_tqdm):
    this_dir = models_dir / str(run)
    this_dir.mkdir(parents=True, exist_ok=True)

    model = MODEL(processor.nr_of_known_tags).to(DEVICE)

    criterion = CRITERION()
    optimizer = OPTIMIZER(model.parameters(), lr=0.001)

    # perform inference run before training
    with torch.no_grad():
        model.eval()

        correct = 0
        total = 0
        initial_preds = []
        initial_targets = []

        for vectors, labels in tqdm(train_loader, "batches", len(train_loader), False):
            vectors = vectors.to(DEVICE).float()
            labels = labels.to(DEVICE).long()

            outputs = model(vectors)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            initial_preds.extend(predictions.cpu().tolist())
            initial_targets.extend(labels.cpu().tolist())

        initial_f1 = f1_score(initial_targets, initial_preds, average="macro", zero_division=0)
        tqdm.write(f"Run {run} | Initial Accuracy: {correct / total:.4f} | Initial F1: {initial_f1:.4f}")

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
            f"Val Acc: {metrics['val_accuracy']:.4f} | "
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
    with open(this_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": test_correct / test_total, "macro_f1": test_macro_f1}, f)
