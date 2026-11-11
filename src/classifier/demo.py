import argparse
import sys
from pathlib import Path
import requests
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from classifier.classifiers import MLP
from classifier.preparation import DataProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fetch_product(barcode: str) -> dict:
    response = requests.get(
        f"https://world.openfoodfacts.org/api/v2/product/{barcode}",
        params={"fields": "categories_tags"},
        headers={
            "user-agent": "dubidubidu"
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") != 1:
        raise ValueError(f"Produkt {barcode} nicht gefunden")

    return data["product"]

model_path = Path(r"C:\Users\Emil\Documents\Repositories\FoodCLS\models\2026_11_11-19_36_19\0\best_model.pt")
processor = DataProcessor.load(model_path.parent.parent / "processor.pkl")
model = MLP(input_features=processor.nr_of_known_tags)
model.load_state_dict(
    torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )
)
model.to(DEVICE)
model.eval()


products = {
    "4061462630392": "Vollmilch",
    "4008400322728": "Kinder Bueno",
    "4061464222502": "Weizenmehl",
    "29084302": "Haferflocken",
    "4061458091640": "Tomatenmark",
}

with torch.inference_mode():
    for barcode, product_name in products.items():
        try:
            product = fetch_product(barcode)
            embedding = processor.process_api(product).float().to(DEVICE)
            category = model.predict(embedding)
            print(f"{product_name}\t:\t{category}")
        except (requests.RequestException, ValueError) as exc:
            print(f"{product_name} ({barcode}) -> Fehler: {exc}")
