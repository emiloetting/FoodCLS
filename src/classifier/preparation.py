import re
import ast
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from collections.abc import Iterable, Mapping
from sklearn.model_selection import train_test_split


class FoodDataset(Dataset):
    """Store product tags and vectorize them with the shared processor on demand."""
    def __init__(self, tags, labels, processor: "DataProcessor"):
        super().__init__()
        processor._check_readiness()
        if len(tags) != len(labels):
            raise ValueError("Tags and labels must have the same length.")
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        self.tags = tags
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        return self.processor.vectorize_tags(self.tags[index]), self.labels[index]

    def _data_split(self, part1:float, part2:float, *, random_state=None) -> tuple[FoodDataset]:
        """Splits object-inherent data using sklearn's train-test-split."""
        assert np.isclose(part1 + part2, 1), "ensure both partition sizes add up to 1."
        indices1, indices2 = train_test_split(
            np.arange(len(self)),
            train_size=part1,
            test_size=part2,
            shuffle=True,
            stratify=self.labels,
            random_state=random_state)
        def subset(indices):
            tags = [self.tags[i] for i in indices]
            return FoodDataset(tags, self.labels[indices], self.processor)
        train_ds = subset(indices1)
        test_ds = subset(indices2)
        return train_ds, test_ds


class DataProcessor():
    def __init__(self):
        self.vocab = None

    @classmethod
    def load(cls, path: str | Path) -> "DataProcessor":
        """Load a processor from a trusted pickle file."""
        with open(path, "rb") as f:
            processor = pickle.load(f)
        if not isinstance(processor, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(processor).__name__}.")
        return processor

    def save(self, path: str | Path) -> None:
        """Save this processor and its complete state as a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _check_readiness(self) -> bool:
        if self.vocab is None:
            raise ValueError("Missing vocab. Ensure loading respective vocab.json.")

    def load_vocab(self, vocab_json:str|Path):
        """Loads vocab. Expects content of `vocab.json` as `list`-type"""
        with open(vocab_json, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.known_tags_dict = {tag: idx for idx, tag in enumerate(self.vocab)}
        self.nr_of_known_tags = len(self.vocab)

    @staticmethod
    def _parse_tags(tags) -> list:
        """Read tag lists, including list strings loaded from CSV."""
        if isinstance(tags, Mapping):
            return []

        if isinstance(tags, str):
            # CSV may contain NumPy lists without commas between quoted tags.
            control_escapes = {code: f"\\x{code:02x}" for code in range(32)}
            values = []
            for match in re.finditer(
                r"'(?:\\.|[^'\\])*'" r'|"(?:\\.|[^"\\])*"', tags
            ):
                try:
                    values.append(ast.literal_eval(match.group(0).translate(control_escapes)))
                except (SyntaxError, ValueError):
                    continue
        elif isinstance(tags, Iterable):
            values = tags
        else:
            return []

        return values

    @staticmethod
    def preprocess_tags(tags) -> list[str]:
        """Normalize tags using the vocabulary extraction rules."""
        cleaned = []
        for value in DataProcessor._parse_tags(tags):
            if not isinstance(value, str):
                continue
            tag = value.lower()
            for old, new in (
                (" - ", ""), ("-", ""), (".", ""), ("|", ""),
                ("ä", "a"), ("ö", "o"), ("ü", "u"),
            ):
                tag = tag.replace(old, new)
            if tag:
                cleaned.append(tag)
        return cleaned

    def normalize_tags(self, tags) -> list[str]:
        """Preprocess raw or normalized tags and keep only known tags."""
        self._check_readiness()
        return [
            tag
            for tag in self.preprocess_tags(tags)
            if tag in self.known_tags_dict
        ]

    def vectorize_tags(self, tags) -> torch.Tensor:
        """Creates onehot-encoded vector of used tags for classification.
        Ignores unknown tags.

        Returns vector full of zeros if no valid tags were found or tags to normalize were [] or `None`"""
        self._check_readiness()
        base = torch.zeros(self.nr_of_known_tags, dtype=torch.float32)
        for p_tag in self.normalize_tags(tags):
            base[int(self.known_tags_dict[p_tag])] = 1       # ignores duplicate tags -> can't get > 1
        return base

    def process_api(self, response_dict: dict) -> torch.Tensor:
        """Vectorize an OFF product dictionary; missing tags produce a zero vector."""
        self._check_readiness()
        return self.vectorize_tags(response_dict.get("categories_tags"))
