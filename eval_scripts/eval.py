"""Unified evaluation harness for LaMI.

Replaces the five per-task scripts. Usage:

    python eval_scripts/eval.py --task imagenetvc \
        --model_name meta-llama/Meta-Llama-3-8B \
        --pretrained_model output/llama3/ft_wiki_laion_220_2 \
        --generate_images --k 10

Supported tasks: imagenetvc, commonsense (--testset {piqa, siqa, hs, winogrande,
arc, obqa, cqa}), boolq, squad, quac.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from operator import itemgetter
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from diffusers import AutoPipelineForText2Image
from nltk import sent_tokenize
from torch.nn.functional import softmax
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from tqdm import tqdm
from transformers import CLIPModel

sys.path.append(os.getcwd())

from models import build_fusion_model, build_tokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_pretrained_weights(model: torch.nn.Module, path: str) -> None:
    state_dict = torch.load(f"{path}.bin", map_location="cpu")
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    model.load_state_dict(state_dict, strict=False)


def build_model(args) -> torch.nn.Module:
    dtype = torch.bfloat16 if "gemma" in args.model_name.lower() else torch.float16
    model = build_fusion_model(args.model_name, torch_dtype=dtype)
    if args.pretrained_model is not None:
        load_pretrained_weights(model, args.pretrained_model)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def maybe_build_diffusion(args, model) -> tuple[Optional[object], Optional[CLIPModel]]:
    if not args.generate_images:
        return None, None
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.sd_model, torch_dtype=model.dtype, low_cpu_mem_usage=False
    ).to(model.device)
    clip = CLIPModel.from_pretrained(args.clip_model).to(model.device)
    return pipe, clip


def build_k_prompts(text: str, k: int, last_n_words: int = 70) -> List[str]:
    """Reproduce the original "shuffle sentences + last-N-words anchor" prompt set."""
    if k <= 1:
        return [" ".join(text.split()[-last_n_words:])]
    base = [text] * (k - 1)
    sentences = [sent_tokenize(p) for p in base]
    variants = [sentences[i - 1][-i % len(sentences[i - 1])] for i in range(1, k)]
    variants.append(" ".join(text.split()[-last_n_words:]))
    return variants


def clip_scores(clip: CLIPModel, processor, images, prompts: List[str]) -> torch.Tensor:
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(clip.device) for k, v in inputs.items()}
    inputs["input_ids"] = inputs["input_ids"][:, -77:]
    inputs["attention_mask"] = inputs["attention_mask"][:, -77:]
    with torch.no_grad():
        out = clip(**inputs)
    image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text_features = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return torch.clamp((image_features * text_features).sum(dim=1), min=0)


def generate_visual_evidence(args, model, pipe, clip, anchor_text: str):
    """Generate k images from ``anchor_text`` and return (pixel_values, scores)."""
    prompts = build_k_prompts(anchor_text, args.k)
    images = pipe(prompts, num_inference_steps=args.sd_steps, guidance_scale=args.sd_guidance).images
    if args.no_clip_rescaling:
        scores = torch.ones(len(images), device=model.device, dtype=model.dtype) * 0.5
    else:
        scores = clip_scores(clip, model.processor, images, prompts) * args.clip_temperature
    pixel_values = model.processor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(model.device, dtype=model.dtype)
    return pixel_values, scores


def append_result(path: str, **fields) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for k, v in fields.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# ImageNetVC
# ---------------------------------------------------------------------------


IMAGENETVC_CANDIDATES = {
    "color": ["brown", "black", "white", "yellow", "green", "gray", "red", "orange", "blue", "silver", "pink"],
    "shape": ["round", "rectangle", "triangle", "square", "oval", "curved", "cylinder", "straight", "cone", "curly", "heart", "star"],
    "material": ["metal", "wood", "plastic", "cotton", "glass", "fabric", "stone", "rubber", "ceramic", "cloth", "leather", "flour", "paper", "clay", "wax", "concrete"],
    "component": ["yes", "no"],
    "others_yes": ["yes", "no"],
    "others_number": ["2", "4", "6", "1", "8", "3", "5"],
    "others_other": ["long", "small", "short", "large", "forest", "water", "ocean", "big", "tree", "ground", "tall", "wild", "outside", "thin", "head", "thick", "circle", "brown", "soft", "land", "neck", "rough", "chest", "smooth", "fur", "hard", "top", "plants", "black", "metal", "books", "vertical", "lake", "grass", "road", "sky", "front", "kitchen", "feathers", "stripes", "baby", "hair", "feet", "mouth", "female", "table"],
}

IMAGENETVC_PROMPTS = [
    "{}",
    "{} Answer:",
    "{} The answer is",
    "Question: {} Answer:",
    "Question: {} The answer is",
]

IMAGENETVC_DEMO_TEMPLATES = [
    "{} {}. ",
    "{} Answer: {}. ",
    "{} The answer is {}. ",
    "Question: {} Answer: {}. ",
    "Question: {} The answer is {}. ",
]


def _imagenetvc_subset(answer: str) -> str:
    if answer in {"yes", "no"}:
        return "others_yes"
    if answer in {"2", "4", "6", "1", "8", "3", "5"}:
        return "others_number"
    return "others_other"


def _load_demonstrations(args, subset: str, prompt_idx: int) -> str:
    df = pd.read_csv(os.path.join(args.imagenetvc_dir, "dev", f"{subset}.csv"), header=0)
    template = IMAGENETVC_DEMO_TEMPLATES[prompt_idx]
    return "".join(template.format(row.question, row.answer) for row in df.itertuples())


def _candidate_token_ids(tokenizer, candidates: List[str]) -> List[int]:
    try:
        token_idx = 1 if tokenizer.add_bos_token else 0
    except AttributeError:
        token_idx = 1
    ids = []
    for cand in candidates + [c.capitalize() for c in candidates]:
        encoded = tokenizer.encode(" " + cand, add_special_tokens=True)
        ids.append(encoded[token_idx if len(encoded) > token_idx else -1])
    return ids


def run_imagenetvc(args) -> None:
    tokenizer = build_tokenizer(args.model_name)
    model = build_model(args)
    pipe, clip = maybe_build_diffusion(args, model)

    subsets = ["color", "shape", "material", "component", "others"]
    overall = {}
    for subset in subsets:
        df = pd.read_csv(os.path.join(args.imagenetvc_dir, f"{subset}.csv"), header=0)
        per_prompt = []
        for prompt_idx, template in enumerate(IMAGENETVC_PROMPTS):
            correct = total = 0
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{subset}/p{prompt_idx}"):
                answer = str(row["answer"]).lower()
                sub = _imagenetvc_subset(answer) if subset == "others" else subset
                candidates = IMAGENETVC_CANDIDATES[sub]
                candidate_ids = _candidate_token_ids(tokenizer, candidates)

                prefix = template.format(row["question"])
                if args.use_icl:
                    prefix = _load_demonstrations(args, sub, prompt_idx) + prefix

                inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
                if pipe is not None:
                    pixel_values, scores = generate_visual_evidence(args, model, pipe, clip, prefix)
                    inputs["pixel_values"] = pixel_values
                    inputs["scores"] = scores

                with torch.no_grad():
                    out = model(**inputs)
                logits = out.logits[:, -1, :].softmax(dim=-1).cpu().mean(dim=0)
                ranked = sorted(
                    ((tokenizer.decode([cid]).strip().lower(), logits[cid].item()) for cid in candidate_ids),
                    key=lambda x: x[1],
                    reverse=True,
                )
                if ranked[0][0] == answer:
                    correct += 1
                total += 1
            per_prompt.append(correct / total)
            print(f"{subset} prompt{prompt_idx}: {per_prompt[-1]:.4f}")
        overall[subset] = per_prompt
        append_result(
            os.path.join(args.output_dir, args.run_name, subset, "results.txt"),
            **{f"prompt{i}": f"{acc:.4f}" for i, acc in enumerate(per_prompt)},
            mean=f"{100 * np.mean(per_prompt):.2f}",
            std=f"{100 * np.std(per_prompt, ddof=1):.2f}",
        )

    print(json.dumps({k: [round(v, 4) for v in vs] for k, vs in overall.items()}, indent=2))


# ---------------------------------------------------------------------------
# CommonsenseQA-style multi-choice
# ---------------------------------------------------------------------------


COMMONSENSE_CONFIG = {
    "piqa": {"dataset": ("piqa", "validation"), "label": "label", "context": ["goal"], "solutions": ["sol1", "sol2"]},
    "siqa": {"dataset": ("lighteval/siqa", "validation"), "label": "label", "context": ["context", "question"], "solutions": ["answerA", "answerB", "answerC"]},
    "hs": {"dataset": ("Rowan/hellaswag", "validation"), "label": "label", "context": ["ctx"], "solutions": "endings"},
    "winogrande": {"dataset": ("winogrande", ["winogrande_xs"], ["validation"]), "label": "answer", "context": ["sentence"], "solutions": ["option1", "option2"]},
    "arc": {"dataset": ("ai2_arc", ["ARC-Easy", "ARC-Challenge"], ["test", "test"]), "label": "answerKey", "context": ["question"], "solutions": "choices"},
    "obqa": {"dataset": ("allenai/openbookqa", ["main"], ["test"]), "label": "answerKey", "context": ["question_stem"], "solutions": "choices"},
    "cqa": {"dataset": ("tau/commonsense_qa", "validation"), "label": "answerKey", "context": ["question"], "solutions": "choices"},
}


def _load_commonsense_dataset(testset: str):
    cfg = COMMONSENSE_CONFIG[testset]["dataset"]
    if isinstance(cfg[1], list):
        parts = [load_dataset(cfg[0], spec)[split] for spec, split in zip(cfg[1], cfg[2])]
        return concatenate_datasets(parts)
    return load_dataset(cfg[0])[cfg[1]]


def _commonsense_query(testset: str, row: Dict, context_cols: List[str], solutions, idx: int) -> str:
    if testset in {"arc", "obqa", "cqa"}:
        return " ".join([row[c] for c in context_cols] + [row[solutions]["text"][idx]])
    if testset == "winogrande":
        return row[context_cols[0]].replace("_", row[solutions[idx]])
    if testset == "hs":
        return row[context_cols[0]] + " " + row[solutions][idx]
    return " ".join([row[c] for c in context_cols] + [row[solutions[idx]]])


def _commonsense_label(label, testset: str) -> int:
    if testset in {"siqa", "winogrande"} or (
        testset == "arc" and isinstance(label, str) and label.isnumeric()
    ):
        return int(label) - 1
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, True: 0, False: 1}
    if label in mapping and label not in {0, 1, 2, 3}:
        return mapping[label]
    return label


def _num_options(row, solutions) -> int:
    if isinstance(solutions, str):
        if solutions == "choices":
            return len(row[solutions]["text"])
        return len(row[solutions])
    return len(solutions)


def run_commonsense(args) -> None:
    if args.testset not in COMMONSENSE_CONFIG:
        raise ValueError(f"Unknown testset '{args.testset}'. Choose from {list(COMMONSENSE_CONFIG)}.")

    tokenizer = build_tokenizer(args.model_name)
    model = build_model(args)
    pipe, clip = maybe_build_diffusion(args, model)
    cfg = COMMONSENSE_CONFIG[args.testset]
    dataset = _load_commonsense_dataset(args.testset)

    correct = total = 0
    for row in tqdm(dataset, desc=args.testset):
        label = _commonsense_label(row[cfg["label"]], args.testset)
        num_options = _num_options(row, cfg["solutions"])

        per_option_loss = []
        for option in range(num_options):
            query = _commonsense_query(args.testset, row, cfg["context"], cfg["solutions"], option)
            if "gemma" in args.model_name.lower():
                inputs = tokenizer(
                    [query], return_tensors="pt", max_length=350, padding="max_length", truncation=True
                )
            else:
                inputs = tokenizer([query], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.data.items()}
            inputs["labels"] = inputs["input_ids"]

            if pipe is not None:
                pixel_values, scores = generate_visual_evidence(args, model, pipe, clip, query)
                inputs["pixel_values"] = pixel_values
                inputs["scores"] = scores

            with torch.no_grad():
                out = model(**inputs)
            per_option_loss.append(torch.exp(out.loss).item())

        pred, _ = min(enumerate(per_option_loss), key=itemgetter(1))
        if pred == int(label):
            correct += 1
        total += 1

    acc = correct / total
    print(f"final accuracy: {acc:.4f}")
    append_result(
        os.path.join(args.output_dir, "results", f"{args.run_name}.txt"),
        model_name=args.model_name, testset=args.testset, k=args.k, accuracy=f"{acc:.2%}",
    )


# ---------------------------------------------------------------------------
# BoolQ
# ---------------------------------------------------------------------------


def _boolq_prompt(row: Dict) -> str:
    return (
        f"Context: {row['passage'].capitalize()}. "
        f"Question: {row['question'].capitalize()}? Answer:"
    )


def run_boolq(args) -> None:
    tokenizer = build_tokenizer(args.model_name)
    model = build_model(args)
    pipe, clip = maybe_build_diffusion(args, model)

    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode(" No", add_special_tokens=False)[0]

    dataset = load_dataset("google/boolq")["validation"]
    label_map = {True: "yes", False: "no"}
    correct = total = 0

    for row in tqdm(dataset, desc="boolq"):
        prompt = _boolq_prompt(row)
        if "gemma" in args.model_name.lower():
            inputs = tokenizer(
                [prompt], return_tensors="pt", max_length=1500, padding="max_length", truncation=True
            )
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.data.items()}

        if pipe is not None:
            pixel_values, scores = generate_visual_evidence(args, model, pipe, clip, prompt)
            inputs["pixel_values"] = pixel_values
            inputs["scores"] = scores

        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        mask = torch.full_like(logits, float("-inf"))
        mask[:, yes_id] = logits[:, yes_id]
        mask[:, no_id] = logits[:, no_id]
        pred_id = softmax(mask, dim=-1).argmax().item()
        pred = tokenizer.decode(pred_id).strip().lower()

        if pred == label_map[row["answer"]]:
            correct += 1
        total += 1

    acc = correct / total
    print(f"BoolQ accuracy: {acc:.3f}")
    append_result(
        os.path.join(args.output_dir, "results", f"{args.run_name or args.model_name.replace('/', '_')}.txt"),
        model_name=args.model_name, testset="boolq", k=args.k, accuracy=f"{acc:.3f}",
    )


# ---------------------------------------------------------------------------
# SQuAD / QuAC (extractive QA via greedy generation)
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gt: str) -> float:
    p_tokens, g_tokens = _normalize(pred).split(), _normalize(gt).split()
    common = Counter(p_tokens) & Counter(g_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(p_tokens)
    recall = same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _max_over_gt(metric, pred, gts) -> float:
    return max(metric(pred, gt) for gt in gts)


def _qa_prompt(title: str, paragraph: str, context: str,
               former_q: str, former_a: str, question: str) -> str:
    return (
        "Answer each question using information in the preceding background paragraph. "
        "If there is not enough information provided, answer with \"Not in background.\"\n\n"
        f"Title: {title}\n"
        + (f"Paragraph: {paragraph}\n\n" if paragraph else "\n")
        + f"Background: {context}\n\n"
        f"Q: {former_q}\n\nA: {former_a}\n\n"
        f"Q: {question}\n\nA:"
    )


def _qa_iter_squad(dataset):
    former_title = ""
    former_q = former_a = None
    for item in dataset:
        if not item["answers"]["text"]:
            continue
        title = item["title"]
        question = item["question"]
        answers = item["answers"]["text"]
        if title == former_title and former_q is not None:
            yield item["title"], "", item["context"], former_q, former_a, question, answers
        former_title = title
        former_q, former_a = question, answers[0]


def _qa_iter_quac(dataset):
    for item in dataset:
        former_q = former_a = None
        title = item["wikipedia_page_title"]
        paragraph = item["section_title"]
        for i, question in enumerate(item["questions"]):
            answer = item["orig_answers"]["texts"][i]
            if answer == "CANNOTANSWER":
                continue
            if former_q is not None:
                yield title, paragraph, item["background"], former_q, former_a, question, [answer]
            former_q, former_a = question, answer


def _generate_answer(args, model, tokenizer, prompt: str, pipe, clip) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    if pipe is not None:
        pixel_values, scores = generate_visual_evidence(args, model, pipe, clip, prompt)
        gen_kwargs.update(pixel_values=pixel_values, scores=scores)
    with torch.no_grad():
        output = model.generate(encoded, **gen_kwargs)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip().split("\n")[0]


def _run_extractive_qa(args, dataset_iter) -> None:
    tokenizer = build_tokenizer(args.model_name)
    model = build_model(args)
    pipe, clip = maybe_build_diffusion(args, model)

    em = f1 = total = 0
    for title, paragraph, context, former_q, former_a, question, answers in tqdm(dataset_iter, desc=args.task):
        prompt = _qa_prompt(title, paragraph, context, former_q, former_a, question)
        pred = _generate_answer(args, model, tokenizer, prompt, pipe, clip)
        em += _max_over_gt(lambda a, b: float(_normalize(a) == _normalize(b)), pred, answers)
        f1 += _max_over_gt(_f1, pred, answers)
        total += 1

    em_pct, f1_pct = 100 * em / total, 100 * f1 / total
    print(json.dumps({"exact_match": em_pct, "f1": f1_pct}))
    append_result(
        os.path.join(args.output_dir, "results", f"{args.run_name or args.model_name.replace('/', '_')}.txt"),
        model_name=args.model_name, testset=args.task, k=args.k,
        exact_match=f"{em_pct:.3f}", f1=f"{f1_pct:.3f}",
    )


def run_squad(args) -> None:
    dataset = load_dataset("rajpurkar/squad_v2")["validation"]
    _run_extractive_qa(args, _qa_iter_squad(dataset))


def run_quac(args) -> None:
    dataset = load_dataset("quac")["validation"]
    _run_extractive_qa(args, _qa_iter_quac(dataset))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


TASKS = {
    "imagenetvc": run_imagenetvc,
    "commonsense": run_commonsense,
    "boolq": run_boolq,
    "squad": run_squad,
    "quac": run_quac,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaMI evaluation harness.")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--model_name", required=True,
                        help="HF model name or local path (used to pick the LaMI variant).")
    parser.add_argument("--pretrained_model", default=None,
                        help="Path (without .bin) to a LaMI checkpoint to load on top of the base model.")
    parser.add_argument("--run_name", default="run")
    parser.add_argument("--output_dir", default="output")

    parser.add_argument("--generate_images", action="store_true",
                        help="Generate k images per prompt and late-fuse them.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--clip_temperature", type=float, default=2.0,
                        help="Scaling on the cosine similarity used as per-image mixing weight.")
    parser.add_argument("--no_clip_rescaling", action="store_true",
                        help="Skip the CLIP-based per-image weighting (equal mixture).")
    parser.add_argument("--sd_model", default="stabilityai/sdxl-turbo")
    parser.add_argument("--sd_steps", type=int, default=1)
    parser.add_argument("--sd_guidance", type=float, default=0.0)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    # ImageNetVC
    parser.add_argument("--imagenetvc_dir", default="evaluation_data/imagenetvc")
    parser.add_argument("--use_icl", action="store_true")

    # Commonsense
    parser.add_argument("--testset", default=None, choices=sorted(COMMONSENSE_CONFIG) + [None])

    # QA
    parser.add_argument("--max_new_tokens", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TASKS[args.task](args)


if __name__ == "__main__":
    main()
