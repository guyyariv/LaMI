# LaMI: Augmenting Large Language Models via Late Multi-Image Fusion

This repo contains the official PyTorch implementation of [*LaMI: Augmenting Large Language Models via Late Multi-Image Fusion*](https://arxiv.org/abs/2406.13621v2) (ACL 2026 Oral).

## Abstract

Commonsense reasoning often requires both textual and visual knowledge, yet Large Language Models (LLMs) trained solely on text lack visual grounding (e.g., "what color is an emperor penguin's belly?"). Visual Language Models (VLMs) perform better on visually grounded tasks but face two limitations: (i) often reduced performance on text-only commonsense reasoning compared to text-trained LLMs, and (ii) adapting newly released LLMs to vision input typically requires costly multimodal training. An alternative augments LLMs with test-time visual signals, improving visual commonsense without harming textual reasoning, but prior designs often rely on early fusion and a single image, which can be suboptimal. We propose a late multi-image fusion method: multiple images are generated from the text prompt with a lightweight parallel sampling, and their prediction probabilities are combined with those of a text-only LLM through a late-fusion layer that integrates projected visual features just before the final prediction. Across visual commonsense and NLP benchmarks, our method significantly outperforms augmented LLMs on visual reasoning, matches VLMs on vision-based tasks, and, when applied to strong LLMs such as LLaMA 3, also improves NLP performance while adding only modest test-time overhead.

<a href="https://arxiv.org/abs/2406.13621v2"><img src="https://img.shields.io/badge/arXiv-2406.13621-b31b1b.svg" height=22.5></a>
<a href="https://guyyariv.github.io/LaMI/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=22.5></a>

## Installation

```bash
git clone git@github.com:guyyariv/LaMI.git
cd LaMI
python -m venv lami
source lami/bin/activate
pip install -r requirements.txt
```

## Supported base models

`gpt2`, `google/gemma-*`, `meta-llama/Meta-Llama-3-*`, and BERT-family MLMs. The training entry point detects the family from `--model_name_or_path` and wraps the stock HuggingFace model with the late-fusion modules.

## Training

The training script wraps any supported base LM with the LaMI late-fusion stack. Only the projection (`mm_proj`) and 1-layer fusion transformer (`fusion_layer`) are trained — the base LM and the CLIP vision encoder stay frozen.

Configure Accelerate once:

```bash
accelerate config
```

Launch training (LLaMA 3 on cropped Visual Genome):

```bash
accelerate launch train.py \
    --run_name llama3_cropped_vg \
    --dataset_name cropped_vg \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --output_dir output/llama3 \
    --per_device_train_batch_size 64 \
    --num_train_epochs 1 \
    --learning_rate 5e-4 \
    --run_bf16 \
    --report_to wandb \
    --with_tracking
```

For full reproducibility, follow the paper's recipe: fine-tune on cropped Visual Genome regions, then continue on WikiText-103 (`--dataset_name wiki`, `--max_elements 200000`) and LAION-220 (`--dataset_name laion_220`).

Data-path flags:

- `--laion_image_root /path/to/COCO` — root for LAION-220 image files (required for `--dataset_name laion_220`).
- `--vg_cache_dir datasets/visual_genome_regions` — where to materialize Visual Genome images that lack a filename.
- `--dataset_cache_dir datasets` — on-disk HuggingFace dataset cache root.

See `python train.py --help` for the full list.

## Evaluation

A single entry point covers all benchmarks:

```bash
python eval_scripts/eval.py \
    --task imagenetvc \
    --model_name meta-llama/Meta-Llama-3-8B \
    --pretrained_model output/llama3/llama3_cropped_vg_0 \
    --generate_images --k 10
```

Supported `--task` values:

| Task | Dataset(s) |
| --- | --- |
| `imagenetvc` | ImageNetVC (color/shape/material/component/others) |
| `commonsense` | PIQA, SIQA, ARC, OBQA, CQA, HellaSwag, WinoGrande (select with `--testset`) |
| `squad` | SQuAD v2 |
| `quac` | QuAC |
| `boolq` | BoolQ |

Common flags:

- `--generate_images` — generate `k` images per prompt with SDXL-Turbo and late-fuse them.
- `--k` — number of generated images per prompt (default `10`).
- `--clip_temperature` — temperature on the CLIP-based per-image weighting (default `2.0`).
- `--no_clip_rescaling` — disable CLIP-score reweighting; weight all images uniformly.

`python eval_scripts/eval.py --help` lists everything.

## Acknowledgments

Built on the [Transformers language-modeling examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) and the [ImageNetVC](https://github.com/hemingkx/ImageNetVC/tree/main) evaluation harness.

## Citation

```bibtex
@inproceedings{yariv-etal-2026-lami,
    title = "{L}a{MI}: Augmenting Large Language Models via Late Multi-Image Fusion",
    author = "Yariv, Guy  and
      Schwartz, Idan  and
      Adi, Yossi  and
      Benaim, Sagie",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-short.3/",
    doi = "10.18653/v1/2026.acl-short.3",
    pages = "18--26",
    ISBN = "979-8-89176-391-3"
}
```

## License

Released under the MIT license. See [LICENSE](LICENSE).
