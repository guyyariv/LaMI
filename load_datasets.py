"""Dataset loaders for LaMI training.

Each loader tokenizes once and caches the resulting HF Dataset under
``--dataset_cache_dir``. Subsequent runs reuse the cached copy.
"""

from __future__ import annotations

import os
from itertools import chain

import datasets
from datasets import load_dataset


def _cache_path(args, name: str) -> str:
    safe = args.model_name_or_path.replace("/", "_")
    os.makedirs(args.dataset_cache_dir, exist_ok=True)
    return os.path.join(args.dataset_cache_dir, f"{name}_{safe}.hf")


def load_laion_220(args, tokenizer, short: bool = False):
    if not args.laion_image_root:
        raise ValueError(
            "load_laion_220 requires --laion_image_root to point at the local image root."
        )

    caption_column = "short_caption" if short else "caption"
    cache = _cache_path(args, "laion_220_short" if short else "laion_220")

    try:
        return datasets.load_from_disk(cache)
    except (FileNotFoundError, OSError):
        pass

    def tokenize(examples):
        text_inputs = tokenizer(
            list(examples[caption_column]),
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        examples["input_ids"] = text_inputs["input_ids"]
        examples["attention_mask"] = text_inputs["attention_mask"]
        examples["image_path"] = [
            os.path.join(args.laion_image_root, url[30:]) for url in examples["url"]
        ]
        return examples

    raw = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS")["train"]
    if args.max_train_samples is not None:
        raw = raw.select(range(min(len(raw), args.max_train_samples)))

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Tokenizing LAION-220",
    )
    tokenized.save_to_disk(cache)
    return tokenized


def load_wiki(args, tokenizer):
    cache = _cache_path(args, f"wiki_{args.wiki_num}_connected")
    try:
        return datasets.load_from_disk(cache)
    except (FileNotFoundError, OSError):
        pass

    raw = load_dataset("wikitext", f"wikitext-{args.wiki_num}-raw-v1")["train"]
    text_col = "text" if "text" in raw.column_names else raw.column_names[0]
    block_size = args.wiki_block_size

    def tokenize(examples):
        return tokenizer(examples[text_col])

    tokenized = raw.map(
        tokenize,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw.column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Tokenizing WikiText",
    )

    def group(examples):
        concat = {k: list(chain(*examples[k])) for k in examples}
        total = (len(concat[next(iter(examples))]) // block_size) * block_size
        result = {k: [v[i : i + block_size] for i in range(0, total, block_size)] for k, v in concat.items()}
        result["labels"] = result["input_ids"].copy()
        result["image_path"] = [None] * len(result["input_ids"])
        return result

    grouped = tokenized.map(
        group,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping WikiText in chunks of {block_size}",
    )
    grouped.save_to_disk(cache)
    return grouped


def load_cropped_vg_regions(args, tokenizer):
    cache = _cache_path(args, "cropped_vg_regions")
    try:
        return datasets.load_from_disk(cache)
    except (FileNotFoundError, OSError):
        pass

    os.makedirs(args.vg_cache_dir, exist_ok=True)

    def tokenize(examples):
        captions, image_paths, crops = [], [], []
        for idx, regions in enumerate(examples["regions"]):
            image = examples["image"][idx]
            try:
                path = image.filename
            except AttributeError:
                path = os.path.join(args.vg_cache_dir, f"{regions[0]['image_id']}.png")
                image.info.pop("icc_profile", None)
                image.save(path)
            for region in regions:
                captions.append(region["phrase"])
                crops.append((region["x"], region["y"], region["width"], region["height"]))
                image_paths.append(path)

        text_inputs = tokenizer(
            captions,
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        examples["input_ids"] = text_inputs["input_ids"]
        examples["attention_mask"] = text_inputs["attention_mask"]
        examples["image_path"] = image_paths
        examples["crop"] = crops
        return examples

    raw = load_dataset("visual_genome", "region_descriptions_v1.2.0")["train"]
    if args.max_train_samples is not None:
        raw = raw.select(range(min(len(raw), args.max_train_samples)))

    tokenized = raw.map(
        tokenize,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw.column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Tokenizing cropped Visual Genome regions",
    )
    tokenized.save_to_disk(cache)
    return tokenized
