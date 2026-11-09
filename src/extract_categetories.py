import duckdb as ddb
from pathlib import Path
import json
from collections import Counter
import pandas as pd

from utils import allowed_tag, get_category_tags, get_main_text


# load configs
with open(Path(__file__).parents[1]/"data"/"config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
db_parquet = Path(config["original_db"])  # extract path to db
tags_csv_pth = Path(config["tags_csv"])

# load data
if not tags_csv_pth.is_file():
    res_df = pd.DataFrame(ddb.sql(
        f"SELECT code, product_name, categories_tags FROM read_parquet('{db_parquet.resolve().as_posix()}')"
    ).df())
    res_df.to_csv(str(tags_csv_pth.resolve()), index=False)
else:
    print("Loading data from csv...")
    res_df = pd.read_csv(
        tags_csv_pth,
        usecols=lambda column: column != "Unnamed: 0",
        dtype={"code": "string"},
    )

# cleanup
res_df["product_name"] = res_df["product_name"].apply(get_main_text)
res_df["categories_tags"] = res_df["categories_tags"].apply(
    lambda value: [tag for tag in get_category_tags(value) if allowed_tag(tag)]
)
res_df = res_df[
    (res_df["product_name"].fillna("") != "")
    & (res_df["categories_tags"].str.len() > 0)
]


# Keep tags used in at least five product rows, counting each tag once per row.
tag_counts = Counter(
    tag
    for tags in res_df["categories_tags"]
    for tag in set(tags)
)
res_df["categories_tags"] = res_df["categories_tags"].apply(
    lambda tags: [tag for tag in tags if tag_counts[tag] >= config["minimum_tag_oocurances"]]
)
res_df = res_df[res_df["categories_tags"].str.len() > 0]
res_df["code"] = res_df["code"].apply(lambda x: str(x))

# find and log all unique tags and store
all_tags = sorted({
    tag
    for categories in res_df["categories_tags"]
    for tag in categories
})

orig_cat_store_pth = Path(config["orig_unique_categories_json"])

with open(str(orig_cat_store_pth.resolve()), "w", encoding="utf-8") as f:
    mapping = {tag: idx for idx, tag in enumerate(all_tags)}
    json.dump(mapping, f, ensure_ascii=False, indent=2)

res_df.to_csv(Path(config["to_LLM_csv"]), index=False)
