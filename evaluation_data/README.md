# Evaluation data

Local CSV / JSONL splits used by the evaluation harness. Most other benchmarks
(BoolQ, SQuAD, QuAC, PIQA, SIQA, ARC, OBQA, CommonsenseQA, HellaSwag,
WinoGrande) are streamed from HuggingFace `datasets` at runtime and are not
checked into this directory.

## Layout

```
evaluation_data/
├── imagenetvc/                  # ImageNetVC test splits
│   ├── color.csv
│   ├── shape.csv
│   ├── material.csv
│   ├── component.csv
│   ├── others.csv
│   └── dev/                     # In-context-learning demonstration pools
│       ├── color.csv
│       ├── shape.csv
│       ├── material.csv
│       ├── component.csv
│       ├── others_yes.csv
│       ├── others_number.csv
│       └── others_other.csv
├── ViComTe/                     # Visual Commonsense Test (color/shape/size pairs)
│   ├── color.jsonl
│   ├── shape.jsonl
│   ├── larger.jsonl
│   └── smaller.jsonl
└── valm/                        # VaLM commonsense splits
    ├── color-concrete-objects.csv
    ├── memory_color_data.csv
    ├── shape.csv
    ├── size.csv
    └── sizePairsFull.txt
```

## Usage

`eval_scripts/eval.py --task imagenetvc` reads `imagenetvc/{subset}.csv` for
each subset and (with `--use_icl`) the matching file under `imagenetvc/dev/`
for demonstration prompts. Override the directory with `--imagenetvc_dir`.

The `ViComTe` and `valm` splits are not consumed by the unified harness in
this revision and are kept here for reference / future ablations.

## Sources

- **ImageNetVC** — Xia et al., 2023. See <https://github.com/hemingkx/ImageNetVC>.
- **ViComTe** — Zhang et al., 2022. See <https://github.com/chenyuheidizhang/vl-commonsense>.
- **VaLM** — Wang et al., 2023. See <https://github.com/microsoft/unilm/tree/master/valm>.
