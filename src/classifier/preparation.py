from torch.utils.data import Dataset
from pathlib import Path
from collections.abc import Iterable, Mapping
import ast
import re
from sklearn.model_selection import train_test_split
import numpy as np
import json
import torch
import pickle


class DataProcessor():
    def __init__(self, allowed_tags_pth:Path):
        with open(str(allowed_tags_pth.resolve()), "r", encoding="utf-8") as f:
            self.known_tags_dict = dict(json.load(f))     # should be dict type, validate later
        self.known_tags_dict_keys = list(self.known_tags_dict.keys())
        self.nr_of_known_tags = len(self.known_tags_dict_keys)

    def normalize_tags(self, tags) -> list[str]:
        """Read bracketed CSV lists and keep normalized, known tags."""
        if tags is None or (isinstance(tags, list) and not tags):
            return []
        if isinstance(tags, str):
            text = tags.strip()
            if not (text.startswith("[") and text.endswith("]")):
                return []
            # NumPy's CSV representation can omit commas between quoted tags.
            control_escapes = {code: f"\\x{code:02x}" for code in range(32)}
            tags = [
                ast.literal_eval(match.group(0).translate(control_escapes))
                for match in re.finditer(
                    r"'(?:\\.|[^'\\])*'" r'|"(?:\\.|[^"\\])*"', text
                )
            ]
        elif not isinstance(tags, Iterable) or isinstance(tags, Mapping):
            return []

        normalized = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            tag = tag.lower().strip()
            if tag.startswith("en:"):
                tag = tag[3:].strip()
            if tag in self.known_tags_dict:
                normalized.append(tag)
        return normalized

    def vectorize_tags(self, tags) -> torch.Tensor:
        """Creates onehot-encoded vector of used tags for classification.
        Ignores unknown tags.
        
        Returns vector full of zeros if no valid tags were found or tags to normalize were [] or `None`"""
        base = torch.zeros(self.nr_of_known_tags, dtype=torch.uint8)
        for p_tag in self.normalize_tags(tags):
            base[int(self.known_tags_dict[p_tag])] = 1       # ignores duplicate tags -> can't get > 1
        return base


    def save_to_pkl(self, pth:Path) -> None:
        """Saves this data processor and its state to a pickle file."""
        with Path(pth).open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load_from_pkl(cls, pth:Path) -> DataProcessor:
        """Loads data processor from pickle-file"""
        with Path(pth).open("rb") as f:
            processor = pickle.load(f)
        if not isinstance(processor, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(processor).__name__}")
        return processor


class FoodDataset(Dataset):
    def __init__(self, onehot_vectors, labels):
        super().__init__()
        if not isinstance(onehot_vectors, torch.Tensor):
            onehot_vectors = torch.tensor(onehot_vectors)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        self.vectors = onehot_vectors
        self.labels = labels

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, index):
        return self.vectors[index], self.labels[index]

    def _data_split(self, part1:float, part2:float) -> tuple[FoodDataset]:
        """Splits object-inherent data using sklearn's train-test-split."""
        assert np.isclose(part1 + part2, 1), "ensure both partition sizes add up to 1."
        X1, X2, y1, y2 = train_test_split(
            self.vectors, 
            self.labels,
            train_size=part1,
            test_size=part2,
            shuffle=True,
            stratify=self.labels)
        X1 = torch.tensor(X1)
        X2 = torch.tensor(X2)
        y1 = torch.tensor(y1)
        y2 = torch.tensor(y2)
        train_ds = FoodDataset(X1, y1)
        test_ds = FoodDataset(X2, y2)
        return train_ds, test_ds
